"""Train PPO (clean observations only) for Agent A (replace) or Agent B (enhance).
Hyperparameters are FIXED a priori (not tuned). Validation scenarios are used only to pick the best
checkpoint (by mean delivery ratio over the frozen eval loads). Writes per-seed: best/final models,
validation curve CSV, training-episode log, metadata JSON."""
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, gymnasium, stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from p2.envs import ReplaceEnv, EnhanceEnv
from p2.sim import make_scenario, VAL_SEED_BASE

CFG = json.loads((ROOT / "configs/phase2_config.json").read_text())
EVAL_LOADS = CFG["eval_loads"]
HP = dict(n_steps=1200, batch_size=300, n_epochs=10, learning_rate=3e-4, gamma=1.0, gae_lambda=0.95,
          clip_range=0.2, ent_coef=0.0, net_arch=[128, 128])
ENVS = {"replace": ReplaceEnv, "enhance": EnhanceEnv}


def val_scenarios(n_per_load):
    return [make_scenario(VAL_SEED_BASE + 1000 + i, L) for L in EVAL_LOADS for i in range(n_per_load)]


def evaluate(model, env_cls, scens):
    env = env_cls(fixed_scenarios=scens); out = []
    for _ in scens:
        obs, _ = env.reset(); done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(a)
        out.append(info["delivery_ratio"])
    return float(np.mean(out))


class ValCB(BaseCallback):
    def __init__(self, env_cls, out, eval_every, scens):
        super().__init__(); self.env_cls, self.out, self.every, self.scens = env_cls, out, eval_every, scens
        self.best, self.rows, self.next = -1.0, [], eval_every; self.ep_rets = []; self.t0 = time.time()
    def _on_step(self):
        for inf in self.locals["infos"]:
            if "delivery_ratio" in inf: self.ep_rets.append(inf["delivery_ratio"])
        if self.num_timesteps >= self.next:
            v = evaluate(self.model, self.env_cls, self.scens)
            tr = float(np.mean(self.ep_rets[-100:])) if self.ep_rets else float("nan")
            self.rows.append((self.num_timesteps, v, tr, round(time.time() - self.t0, 1)))
            if v > self.best: self.best = v; self.model.save(str(self.out / "best"))
            self.next += self.every
        return True


def train_one(agent, seed, steps, eval_every, out_root, n_val):
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    out = out_root / agent / f"seed{seed}"; out.mkdir(parents=True, exist_ok=True)
    env_cls = ENVS[agent]; env = env_cls(load_range=tuple(CFG["train_load_range"]))
    pk = dict(net_arch=dict(pi=HP["net_arch"], vf=HP["net_arch"]))
    if agent == "enhance": pk["log_std_init"] = -1.0
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, policy_kwargs=pk,
                **{k: HP[k] for k in ["n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef"]})
    cb = ValCB(env_cls, out, eval_every, val_scenarios(n_val)); t0 = time.time()
    v0 = evaluate(model, env_cls, cb.scens); cb.rows.append((0, v0, float("nan"), 0.0))
    model.learn(total_timesteps=steps, callback=cb); wall = time.time() - t0
    model.save(str(out / "final"))
    with open(out / "val_curve.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["timesteps", "val_delivery_ratio", "train_delivery_ratio_last100", "wall_s"]); w.writerows(cb.rows)
    flat = torch.cat([p.flatten() for p in model.policy.parameters()]).detach().numpy()
    meta = dict(agent=agent, train_seed=seed, total_timesteps=steps, eval_every=eval_every, hyperparameters=HP,
                best_val=cb.best, final_val=cb.rows[-1][1], init_val=v0, wall_seconds=round(wall, 1),
                final_param_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
                versions=dict(python=platform.python_version(), torch=torch.__version__, sb3=sb3.__version__, gymnasium=gymnasium.__version__, numpy=np.__version__))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{agent} seed={seed} steps={steps} init_val={v0:.4f} best_val={cb.best:.4f} final_val={cb.rows[-1][1]:.4f} wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=ENVS, required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=300_000); ap.add_argument("--eval-every", type=int, default=24_000)
    ap.add_argument("--n-val", type=int, default=25, help="validation scenarios per eval load")
    ap.add_argument("--out", type=Path, default=ROOT / "results/models")
    a = ap.parse_args()
    for s in a.seeds: train_one(a.agent, s, a.steps, a.eval_every, a.out, a.n_val)
