"""Paired evaluation of shortest-path, greedy-telemetry and PPO under poisoning.

For every (training seed, scenario seed) the SAME demand sequence is used for
all conditions (scenario = ``env.reset(seed=scenario_seed)``), so comparisons are
paired. Writes one row per episode to ``results/episodes.csv`` and the full
configuration to ``results/config.json``.

Per-episode metrics (all on TRUE loads, final state of the episode):
  max_utilization      max_e L_e / C                        (lower is better)
  mean_utilization     mean_e L_e / C                       (lower is better)
  squared_utilization  mean_e (L_e / C)^2                   (lower is better)
  episode_reward       sum_t -mean_e (L_e(t)/C)^2           (higher is better; a
                       congestion-shaped training signal, NOT a security metric)
  overloaded_links     #{e : L_e / C > 1}                   (lower is better)
  total_hops           sum_t (|path_t| - 1)                 (unit-weight path cost;
                       lower is better)
  invalid_actions      count of out-of-range actions        (always 0: the env
                       raises on invalid actions and PPO's Discrete head cannot emit them)
Attack / decision metrics (PPO rows):
  traj_action_changes  # steps where the action differs from the same model's
                       delta=0 action in the same scenario (compounds after the
                       first divergence because states diverge)
  state_flips          # steps where, on the CLEAN trajectory's own states, the
                       action under poisoned observation differs from the action
                       under clean observation (no trajectory divergence)
  pert_mean_abs        mean |obs_poisoned - obs_clean| over the poisoned link
                       entries and over steps (the ACTUAL perturbation)
  pert_active_frac     fraction of steps where >=1 poisoned entry actually changed
                       after clipping (<=1; saturation makes poison a no-op)
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from src.policies import greedy_telemetry_action, shortest_path_action  # noqa: E402
from src.routing_env import DEFAULT_POISON_EDGES, N_DEMANDS, RoutingEnv  # noqa: E402

DEFAULT_DELTAS = [0.0, 0.10, 0.25, 0.50, 1.00]
LEGACY_SCENARIOS = list(range(100, 120))  # the 20 scenarios used by the original scripts


def rollout(env, action_fn, scenario_seed):
    """One episode. Returns (final_info, actions, perturbation stats)."""
    obs, _ = env.reset(seed=scenario_seed)
    n = env.n_edges
    actions, abs_diffs, active = [], [], []
    info = {}
    done = False
    while not done:
        diff = np.abs(env.last_obs[:n] - env.last_clean_obs[:n])[list(env.poison_edges)]
        abs_diffs.append(float(diff.mean()))
        active.append(bool(diff.max() > 0))
        a = int(action_fn(env, obs))
        actions.append(a)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return info, actions, float(np.mean(abs_diffs)), float(np.mean(active))


def clean_trajectory_flips(model, deltas, scenario_seed):
    """State-level counterfactual: follow the CLEAN PPO trajectory; at each visited
    state compare the action under clean vs poisoned observation."""
    env = RoutingEnv(poison=False)
    obs, _ = env.reset(seed=scenario_seed)
    flips = {d: 0 for d in deltas}
    done = False
    while not done:
        clean_a = int(model.predict(obs, deterministic=True)[0])
        for d in deltas:
            if d == 0.0:
                continue
            a = int(model.predict(env.counterfactual_observation(d), deterministic=True)[0])
            flips[d] += int(a != clean_a)
        obs, _, term, trunc, _ = env.step(clean_a)
        done = term or trunc
    env.close()
    return flips


def ppo_fn(model):
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=ROOT / "results" / "models")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--scenario-seeds", type=int, nargs="+", default=list(range(100, 200)))
    ap.add_argument("--deltas", type=float, nargs="+", default=DEFAULT_DELTAS)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    deltas = sorted(set(args.deltas) | {0.0})

    rows = []
    t0 = time.time()

    def base(method, delta, model_seed, scen, info, actions, pmean, pact):
        return {
            "method": method, "delta": delta, "train_seed": model_seed,
            "scenario_seed": scen, "legacy_scenario": scen in LEGACY_SCENARIOS,
            **info, "invalid_actions": 0, "pert_mean_abs": pmean,
            "pert_active_frac": pact, "actions": json.dumps(actions),
        }

    # --- reference policies (no trained model): train_seed = -1
    for scen in args.scenario_seeds:
        env = RoutingEnv(poison=False)
        info, acts, pm, pa = rollout(env, shortest_path_action, scen)
        rows.append(base("shortest_path", 0.0, -1, scen, info, acts, pm, pa))
        # SP does not read observations, so a poisoned run is verified, not assumed:
        for d in [x for x in deltas if x > 0]:
            envp = RoutingEnv(poison=True, delta=d)
            infop, actsp, pmp, pap = rollout(envp, shortest_path_action, scen)
            rows.append(base("shortest_path", d, -1, scen, infop, actsp, pmp, pap))
        for d in deltas:
            envg = RoutingEnv(poison=d > 0, delta=d)
            infog, actsg, pmg, pag = rollout(envg, greedy_telemetry_action, scen)
            rows.append(base("greedy_telemetry", d, -1, scen, infog, actsg, pmg, pag))
    print(f"reference policies done ({time.time() - t0:.0f}s)", flush=True)

    # --- PPO, one trained model per training seed
    for ts in args.train_seeds:
        model = PPO.load(str(args.models_dir / f"ppo_seed{ts}.zip"), device="cpu")
        fn = ppo_fn(model)
        for scen in args.scenario_seeds:
            clean_actions = None
            flips = clean_trajectory_flips(model, deltas, scen)
            for d in deltas:
                env = RoutingEnv(poison=d > 0, delta=d)
                info, acts, pm, pa = rollout(env, fn, scen)
                if d == 0.0:
                    clean_actions = acts
                row = base("ppo", d, ts, scen, info, acts, pm, pa)
                row["traj_action_changes"] = int(sum(x != y for x, y in zip(clean_actions, acts)))
                row["state_flips"] = 0 if d == 0.0 else flips[d]
                row["n_decisions"] = len(acts)
                rows.append(row)
        print(f"model {ts} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "episodes.csv", index=False)

    config = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_seeds": args.train_seeds,
        "scenario_seeds": [args.scenario_seeds[0], args.scenario_seeds[-1], len(args.scenario_seeds)],
        "scenario_seed_list_is_contiguous_range": args.scenario_seeds
        == list(range(args.scenario_seeds[0], args.scenario_seeds[-1] + 1)),
        "legacy_scenario_seeds": [LEGACY_SCENARIOS[0], LEGACY_SCENARIOS[-1]],
        "deltas": deltas,
        "poison": {"type": "additive_clip", "links": list(DEFAULT_POISON_EDGES),
                   "edges": [[0, 1], [1, 3]], "note": "delta added to observed utilization, clipped to [0,1]"},
        "n_demands_per_episode": N_DEMANDS,
        "policy_inference": "deterministic=True",
        "paired_design": "identical scenario_seed -> identical demand sequence for every method/delta/model",
        "n_rows": int(len(df)),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__, "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"wrote {len(df)} rows -> {args.out / 'episodes.csv'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
