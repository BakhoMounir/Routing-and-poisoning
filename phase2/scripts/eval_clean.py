"""Clean evaluation on the FROZEN test scenarios (seeds TEST_SEED_BASE+i) x eval loads.
Paired by (scenario_seed, load) across every method and model. No attack is run here.
Also runs a telemetry ABLATION (utilization features zeroed) purely to test whether policies use telemetry."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
from stable_baselines3 import PPO
from p2.sim import Network, Episode, make_scenario, TEST_SEED_BASE
from p2.baselines import BASELINES
from p2.envs import ReplaceEnv, EnhanceEnv, Telemetry, build_features

CFG = json.loads((ROOT / "configs/phase2_config.json").read_text())
ENVS = {"replace": ReplaceEnv, "enhance": EnhanceEnv}


class ZeroTelemetry(Telemetry):          # diagnostic ablation: blind the agent to utilization
    def observe(self, u): return np.zeros_like(u)


def run_baseline(net, name, scen):
    ep = Episode(net, scen); fn = BASELINES[name]; same0 = 0
    while not ep.done:
        s, d, _ = ep.demand(); e = fn(net, ep, ep.true_util())
        same0 += int(np.array_equal(e, net.candidates(s, d)[0][1])); ep.route(e)
    m = ep.metrics(); m["frac_route_is_minhop_cand0"] = same0 / len(scen.vol); return m


def run_ppo(net, agent, model, scen, telemetry=None):
    env = ENVS[agent](net=net, telemetry=telemetry); obs, _ = env.reset(options={"scenario": scen}); done = False; same0 = 0; idx = []
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        s, d, _ = env.ep.demand(); e = env._edges_for(a)
        same0 += int(np.array_equal(e, net.candidates(s, d)[0][1]))
        if agent == "replace": idx.append(int(a))
        obs, _, done, _, info = env.step(a)
    info = dict(info); info["frac_route_is_minhop_cand0"] = same0 / len(scen.vol)
    if idx: info["action_hist"] = json.dumps(np.bincount(idx, minlength=4).tolist())
    return info


def state_flip_rate(net, agent, model, scen):
    """Along the CLEAN trajectory: fraction of decisions whose ROUTE changes if utilization features are zeroed."""
    env = ENVS[agent](net=net); obs, _ = env.reset(options={"scenario": scen}); done = False; n = ch = 0
    wc = agent == "replace"
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        blind = build_features(net, env.ep, np.zeros(net.E), wc); b, _ = model.predict(blind, deterministic=True)
        ch += int(not np.array_equal(env._edges_for(a), env._edges_for(b))); n += 1
        obs, _, done, _, _ = env.step(a)
    return ch / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=ROOT / "results/models"); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--n-test", type=int, default=200); ap.add_argument("--n-ablate", type=int, default=50)
    ap.add_argument("--ckpt", default="best"); ap.add_argument("--out", type=Path, default=ROOT / "results/clean")
    a = ap.parse_args(); torch.set_num_threads(1); a.out.mkdir(parents=True, exist_ok=True)
    net = Network(); loads = CFG["eval_loads"]; t0 = time.time(); rows = []; abl = []
    scens = [(i, L, make_scenario(TEST_SEED_BASE + i, L)) for L in loads for i in range(a.n_test)]
    for name in BASELINES:
        for i, L, sc in scens:
            rows.append(dict(method=name, agent="baseline", model_seed=-1, scenario_idx=i, scenario_seed=sc.seed, load=L, **run_baseline(net, name, sc)))
    print(f"baselines done {time.time() - t0:.0f}s", flush=True)
    for agent in ENVS:
        for ms in a.seeds:
            model = PPO.load(str(a.models / agent / f"seed{ms}" / a.ckpt), device="cpu")
            for i, L, sc in scens:
                rows.append(dict(method=f"ppo_{agent}", agent=agent, model_seed=ms, scenario_idx=i, scenario_seed=sc.seed, load=L, **run_ppo(net, agent, model, sc)))
            for L in loads:                                         # ablation on a subset of the same test scenarios
                f, dl, sw = [], [], []
                for i in range(a.n_ablate):
                    sc = make_scenario(TEST_SEED_BASE + i, L)
                    clean = run_ppo(net, agent, model, sc); blind = run_ppo(net, agent, model, sc, ZeroTelemetry())
                    dl.append(blind["delivery_ratio"] - clean["delivery_ratio"]); f.append(state_flip_rate(net, agent, model, sc))
                abl.append(dict(agent=agent, model_seed=ms, load=L, n=a.n_ablate, state_route_change_when_util_zeroed=float(np.mean(f)), delivery_change_when_util_zeroed=float(np.mean(dl))))
            print(f"{agent} seed {ms} done {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "episodes_clean.csv", index=False); pd.DataFrame(abl).to_csv(a.out / "telemetry_ablation.csv", index=False)
    (a.out / "eval_config.json").write_text(json.dumps(dict(test_seed_base=TEST_SEED_BASE, n_test=a.n_test, loads=loads, ckpt=a.ckpt, model_seeds=a.seeds, n_ablate=a.n_ablate, rows=len(rows), wall_s=round(time.time() - t0)), indent=2))
    print("wrote", len(rows), "rows")
