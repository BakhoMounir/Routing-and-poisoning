"""Stage D: resumable long training of Agent A with a predeclared validation-based stopping rule.

Pure helpers (`plateau_stop`, `select_best`) need no torch. `train_agent_a` imports torch/SB3 lazily and is only
called from scripts/x_04_train_agent_a.py with --run. Same PPO hyper-parameters as the legacy runs (configs/x_experiments.json::ppo_hp).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import atomic_write_text


def smoothed(vals, smooth):
    v = np.asarray(vals, float)
    return np.array([v[max(0, k - smooth + 1): k + 1].mean() for k in range(len(v))])


def plateau_stop(vals, smooth, window, min_evals, min_improvement):
    """vals = raw validation scores at evals 1..K. Returns (stop?, info). Stops from eval K >= min_evals when the
    smoothed score gained less than `min_improvement` over the last `window` evals."""
    K = len(vals)
    if K < max(min_evals, smooth + window):
        return False, {"k": K, "gain": None}
    S = smoothed(vals, smooth)
    gain = float(S[-1] - S[-1 - window])
    return gain < min_improvement, {"k": K, "gain": gain}


def select_best(vals):
    """Index (0-based) of the best raw validation score; earliest on ties."""
    return int(np.argmax(np.asarray(vals, float)))



def check_resume_checkpoint_pair(state_path, checkpoint_path):
    """Return fresh/resume, refusing to guess when only half a checkpoint pair exists."""
    state_path, checkpoint_path = Path(state_path), Path(checkpoint_path)
    has_state, has_checkpoint = state_path.is_file(), checkpoint_path.is_file()
    if has_state != has_checkpoint:
        raise RuntimeError(
            "incomplete training recovery pair: state.json and last.zip must both exist; "
            "preserve the files and inspect before resuming"
        )
    return "resume" if has_state else "fresh"


def validate_resume_state(state, seed, checkpoint_step):
    """Validate persisted training state against its checkpoint before resuming."""
    import math

    required = {
        "seed", "resumes", "init_val", "history", "next_eval",
        "best_val", "best_step", "wall_s", "ep_rets", "stop_reason",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("invalid training resume state: missing required state fields")

    if state["seed"] != seed:
        raise ValueError(
            f"training resume seed mismatch: state seed={state['seed']}, requested seed={seed}"
        )

    if type(state["resumes"]) is not int or state["resumes"] < 0:
        raise ValueError("invalid training resume state: resumes must be a nonnegative integer")

    if type(state["next_eval"]) is not int or state["next_eval"] <= 0:
        raise ValueError("invalid training resume state: next_eval must be a positive integer")

    def finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    if not finite_number(state["init_val"]):
        raise ValueError("invalid training resume state: init_val must be finite")
    if not finite_number(state["best_val"]):
        raise ValueError("invalid training resume state: best_val must be finite")
    if not finite_number(state["wall_s"]) or state["wall_s"] < 0:
        raise ValueError("invalid training resume state: wall_s must be finite and nonnegative")

    if state["stop_reason"] not in (None, "plateau"):
        raise ValueError("invalid training resume state: unexpected stop_reason")

    history = state["history"]
    if not isinstance(history, list):
        raise ValueError("invalid training resume state: history must be a list")

    previous_step = 0
    best_history_val = -math.inf
    best_history_step = None

    for item in history:
        if not isinstance(item, dict):
            raise ValueError("invalid training resume state: history entries must be objects")

        step = item.get("step")
        value = item.get("val")
        if type(step) is not int or step <= previous_step:
            raise ValueError("invalid training resume state: history steps must increase strictly")
        if not finite_number(value):
            raise ValueError("invalid training resume state: history val must be finite")

        if value > best_history_val:
            best_history_val = value
            best_history_step = step

        for key in ("per_load",):
            if not isinstance(item.get(key), dict) or not item[key]:
                raise ValueError(f"invalid training resume state: {key} must be a nonempty object")

        for value_at_load in item["per_load"].values():
            if not finite_number(value_at_load):
                raise ValueError("invalid training resume state: per_load values must be finite")

        for key in ("wall_s",):
            if not finite_number(item.get(key)) or item[key] < 0:
                raise ValueError(f"invalid training resume state: history {key} must be finite and nonnegative")

        train_last100 = item.get("train_last100")
        if train_last100 is not None and not finite_number(train_last100):
            raise ValueError("invalid training resume state: train_last100 must be finite or None")

        previous_step = step

    if history:
        if history[-1]["step"] != checkpoint_step:
            raise ValueError(
                f"training resume step mismatch: latest history step={history[-1]['step']}, "
                f"checkpoint step={checkpoint_step}"
            )
        if state["best_step"] != best_history_step or not math.isclose(
            state["best_val"], best_history_val, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("invalid training resume state: best_val/best_step disagree with history")
    elif checkpoint_step != 0 or state["best_step"] is not None or state["best_val"] != -1.0:
        raise ValueError("invalid training resume state: empty history disagrees with checkpoint or best score")

    if type(state["best_step"]) is not int and state["best_step"] is not None:
        raise ValueError("invalid training resume state: best_step must be an integer or None")

    if not isinstance(state["ep_rets"], list) or len(state["ep_rets"]) > 100:
        raise ValueError("invalid training resume state: ep_rets must be a list of at most 100 values")
    if any(not finite_number(value) for value in state["ep_rets"]):
        raise ValueError("invalid training resume state: ep_rets values must be finite")

    return True

def train_agent_a(cfg, seed, out_dir, val_scens, log=print):
    import platform
    import torch
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from p2.envs import ReplaceEnv
    from p2.sim import Network
    from .rollout import evaluate_fast, param_sha256

    out_dir = Path(out_dir)
    if (out_dir / "meta.json").is_file():
        log(f"[skip] seed {seed}: meta.json exists (complete)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    t, hp = cfg["train_agent_a"], cfg["ppo_hp"]
    st_cfg = t["stopping"]
    torch.set_num_threads(1)
    net = Network()
    env = ReplaceEnv(load_range=tuple(cfg["train_load_range"]))
    pk = dict(net_arch=dict(pi=hp["net_arch"], vf=hp["net_arch"]))
    keys = ["n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef"]
    state_p, last_p = out_dir / "state.json", out_dir / "last.zip"
    recovery = check_resume_checkpoint_pair(state_p, last_p)
    if recovery == "resume":
        try:
            state = json.loads(state_p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read training recovery state {state_p}; preserve files") from exc
        model = PPO.load(str(last_p), env=env, device="cpu")
        validate_resume_state(state, seed=seed, checkpoint_step=int(model.num_timesteps))
        state["resumes"] += 1
        model.set_random_seed(seed + 100_000 * state["resumes"])
        log(f"[resume] seed {seed} from step {model.num_timesteps} (resume #{state['resumes']}; statistically, not bit-wise, equivalent)")
    else:
        torch.manual_seed(seed); np.random.seed(seed)
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, policy_kwargs=pk, **{k: hp[k] for k in keys})
        v0 = float(np.mean(evaluate_fast(net, "replace", model, [s for _, _, s in val_scens])))
        state = dict(seed=seed, init_val=v0, history=[], best_val=-1.0, best_step=None, next_eval=t["eval_every"],
                     wall_s=0.0, resumes=0, stop_reason=None, ep_rets=[])
        log(f"[start] seed {seed} init_val={v0:.4f}")
    t0, wall0 = time.time(), state["wall_s"]
    scen_only = [s for _, _, s in val_scens]
    loads = sorted({L for _, L, _ in val_scens})

    class Val(BaseCallback):
        def _on_step(self):
            for inf in self.locals["infos"]:
                if "delivery_ratio" in inf:
                    state["ep_rets"] = (state["ep_rets"] + [inf["delivery_ratio"]])[-100:]
            if self.num_timesteps < state["next_eval"]:
                return True
            r = evaluate_fast(net, "replace", self.model, scen_only)
            v = float(np.mean(r))
            per_load = {str(L): float(np.mean([x for x, (_, l, _) in zip(r, val_scens) if l == L])) for L in loads}
            state["wall_s"] = wall0 + time.time() - t0
            state["history"].append(dict(step=int(self.num_timesteps), val=v, per_load=per_load,
                                         train_last100=float(np.mean(state["ep_rets"])) if state["ep_rets"] else None,
                                         wall_s=round(state["wall_s"], 1)))
            if v > state["best_val"]:
                state["best_val"], state["best_step"] = v, int(self.num_timesteps)
                self.model.save(str(out_dir / "best"))
            self.model.save(str(out_dir / "last"))
            state["next_eval"] += t["eval_every"]
            stop, info = plateau_stop([h["val"] for h in state["history"]], st_cfg["smooth_evals"], st_cfg["window_evals"],
                                      st_cfg["min_evals"], st_cfg["min_improvement"])
            if stop:
                state["stop_reason"] = "plateau"
            atomic_write_text(state_p, json.dumps(state, indent=1))
            log(f"  seed {seed} step {self.num_timesteps}: val={v:.4f} best={state['best_val']:.4f} gain={info['gain']}")
            if stop:
                return False
            return True

    remaining = t["max_steps"] - model.num_timesteps
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=Val(), reset_num_timesteps=False)
    vals = [h["val"] for h in state["history"]]
    if state["stop_reason"] is None:
        state["stop_reason"] = "budget_exhausted"
    stop, info = plateau_stop(vals, st_cfg["smooth_evals"], st_cfg["window_evals"], 0, -1e9)
    gain_end = info["gain"]
    model.save(str(out_dir / "final"))
    meta = dict(agent="replace", run_id=t["run_id"], train_seed=seed, max_steps=t["max_steps"], steps_trained=int(model.num_timesteps),
                eval_every=t["eval_every"], stop_reason=state["stop_reason"], still_improving_at_end=bool(gain_end is not None and gain_end >= st_cfg["min_improvement"]),
                smoothed_gain_last_window=gain_end, init_val=state["init_val"], best_val=state["best_val"], best_step=state["best_step"],
                final_val=vals[-1] if vals else None, val_range=t["val_range"], n_val_per_load=t["n_val_per_load"],
                hyperparameters=hp, stopping_rule=st_cfg, resumes=state["resumes"], wall_seconds=round(state["wall_s"], 1),
                final_param_sha256=param_sha256(model), config_sha256=cfg["_sha256"],
                versions=dict(python=platform.python_version(), torch=torch.__version__, sb3=sb3.__version__, numpy=np.__version__),
                claim="No convergence is claimed. 'plateau' = predeclared rule fired; 'budget_exhausted' = rule did not fire within max_steps.")
    atomic_write_text(out_dir / "val_curve.csv", "step,val,train_last100,wall_s\n" + "\n".join(
        f"{h['step']},{h['val']},{h['train_last100']},{h['wall_s']}" for h in state["history"]) + "\n")
    atomic_write_text(out_dir / "meta.json", json.dumps(meta, indent=2))      # written LAST = completion marker
    log(f"[done] seed {seed}: best_val={state['best_val']:.4f} at step {state['best_step']} stop={state['stop_reason']}")
