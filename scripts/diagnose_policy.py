"""Why is the attack effect small? Evidence on competing explanations.

On the states visited by each model's CLEAN rollouts (scenarios 100-199) measure:
  frac_action0            share of decisions choosing the shortest-hop path
  abl_all_util            share of decisions that change if ALL 13 utilization
                          features are set to 0 (does the policy use telemetry at all?)
  abl_poisoned_links      ... if only the 2 poisoned links' features are set to 0
  abl_other_links         ... if only the other 11 links' features are set to 0
  exposure_chosen         share of decisions whose CHOSEN path crosses a poisoned link
  exposure_any_candidate  share of decisions where ANY candidate path crosses one
  poison_at_max           share of states where the poisoned-link features are already
                          saturated at 1.0 (so +delta is a no-op)
Output: results/policy_diagnostics.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from src.routing_env import RoutingEnv  # noqa: E402


def collect_states(model, scenarios):
    env = RoutingEnv(poison=False)
    obs_list, act, chosen_exposed, any_exposed = [], [], [], []
    pe = set(env.poison_edges)
    for s in scenarios:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            a = int(model.predict(obs, deterministic=True)[0])
            obs_list.append(obs.copy())
            act.append(a)
            chosen_exposed.append(bool(pe & set(env.path_edge_indices(env.paths[a]))))
            any_exposed.append(any(pe & set(env.path_edge_indices(p)) for p in env.paths))
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    return np.array(obs_list), np.array(act), np.array(chosen_exposed), np.array(any_exposed), env


def main():
    torch.set_num_threads(1)
    rows = []
    for ts in range(42, 52):
        model = PPO.load(str(ROOT / "results" / "models" / f"ppo_seed{ts}.zip"), device="cpu")
        X, a, ce, ae, env = collect_states(model, range(100, 200))
        n = env.n_edges
        pe = list(env.poison_edges)
        others = [i for i in range(n) if i not in pe]

        def changed(mod):
            Z = X.copy()
            mod(Z)
            b, _ = model.predict(Z, deterministic=True)
            return float((b != a).mean())

        rows.append(dict(
            train_seed=ts, n_decisions=len(a), frac_action0=float((a == 0).mean()),
            abl_all_util=changed(lambda Z: Z.__setitem__((slice(None), slice(0, n)), 0.0)),
            abl_poisoned_links=changed(lambda Z: Z.__setitem__((slice(None), pe), 0.0)),
            abl_other_links=changed(lambda Z: Z.__setitem__((slice(None), others), 0.0)),
            exposure_chosen=float(ce.mean()), exposure_any_candidate=float(ae.mean()),
            poison_at_max=float((X[:, pe] >= 1.0).all(axis=1).mean()),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "policy_diagnostics.csv", index=False)
    print(df.round(3).to_string(index=False))
    print("\nmean across models:\n", df.drop(columns="train_seed").mean().round(3).to_string())


if __name__ == "__main__":
    main()
