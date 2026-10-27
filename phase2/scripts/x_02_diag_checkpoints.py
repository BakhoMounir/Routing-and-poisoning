#!/usr/bin/env python3
"""Stage B - READ-ONLY diagnostics of SHIPPED checkpoints on the development set val_v2 (minutes, CPU, needs torch+SB3). No training.
Agent B (enhance): policy std, action saturation, edge-weight distribution, state-dependence of the output, and how often the chosen route differs
from the unit-weight-Dijkstra route at the SAME state; accepted/offered ratio vs the unit-weight null.  [Issue 2: unverified from the dump]
Agent A (replace): action histogram, policy confidence/entropy, fraction of decisions on candidate 0, ratio vs min-hop / shortest_feasible_k.
Interpretation is EXPLORATORY. Statistical CIs are scenario-sampling only (per model)."""
import os
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from p2x import cli
from p2x.cli import log
from p2x.config import atomic_write_text, results_dir, scenarios
from p2x.manifest import RunManifest
from p2x.policies import build_baselines
from p2x.rollout import ENVS, load_model, run_baseline
from p2x.stats import summarize_diff

W_LN = float(np.log(10.0))


def probe(model, obs, agent):
    """Raw (unclipped) deterministic action for Agent B, or action probabilities for Agent A. `model.diag_probe` overrides (tests)."""
    if hasattr(model, "diag_probe"):
        return model.diag_probe(obs)
    import torch
    with torch.no_grad():
        ot, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(ot)
        return (dist.get_actions(deterministic=True) if agent == "enhance" else dist.distribution.probs).cpu().numpy()[0]


def policy_std(model):
    if hasattr(model, "diag_std"):
        return np.asarray(model.diag_std())
    return np.exp(model.policy.log_std.detach().cpu().numpy())


def diagnose(net, agent, model, scens, unit_fn):
    env = ENVS[agent](net=net); ratios, raw, ent, conf, hist = [], [], [], [], np.zeros(4)
    same_unit = same_c0 = nonmin = n = 0; wmin, wmax, ratio_w = [], [], []
    for _, _, sc in scens:
        obs, _ = env.reset(options={"scenario": sc}); done, info = False, {}
        while not done:
            a, _ = model.predict(obs, deterministic=True); o = probe(model, obs, agent); s, d, _ = env.ep.demand()
            e = np.asarray(env._edges_for(a)); c0 = net.candidates(s, d)[0][1]
            same_c0 += int(np.array_equal(e, c0)); nonmin += int(len(e) > len(c0)); n += 1
            if agent == "enhance":
                raw.append(o); w = np.exp(np.clip(o, -1, 1) * W_LN); wmin.append(w.min()); wmax.append(w.max()); ratio_w.append(w.max() / w.min())
                same_unit += int(np.array_equal(e, unit_fn(net, env.ep, None)))
            else:
                p = np.clip(o, 1e-12, 1); ent.append(float(-(p * np.log(p)).sum())); conf.append(float(p.max())); hist[int(a)] += 1
            obs, _, done, _, info = env.step(a)
        ratios.append(info["delivery_ratio"])
    r = dict(frac_route_is_minhop_cand0=same_c0 / n, frac_nonminimal_hops=nonmin / n)
    if agent == "enhance":
        R = np.asarray(raw)
        r.update(frac_route_equals_unit_dijkstra=same_unit / n, frac_components_outside_bounds=float((np.abs(R) > 1).mean()), frac_components_ge_0p99=float((np.abs(R) >= 0.99).mean()),
                 raw_action_abs_mean=float(np.abs(R).mean()), raw_action_state_std_mean=float(R.std(axis=0).mean()),
                 weight_min_mean=float(np.mean(wmin)), weight_max_mean=float(np.mean(wmax)), weight_max_over_min_median=float(np.median(ratio_w)),
                 frac_decisions_weights_within_5pct=float(np.mean(np.asarray(ratio_w) < 1.05)), policy_std_mean=float(policy_std(model).mean()),
                 policy_std_min=float(policy_std(model).min()), policy_std_max=float(policy_std(model).max()))
    else:
        r.update(policy_entropy_mean=float(np.mean(ent)), policy_max_prob_mean=float(np.mean(conf)), action_hist=json.dumps((hist / hist.sum()).round(4).tolist()))
    return np.asarray(ratios), r


def execute(cfg, a, net, sha, loader=load_model):
    d = cfg["diagnostics"]; n = a.n_per_load or d["n_per_load"]; loads = cfg["loads"]
    out = cli.prepare_out(a.out or results_dir(cfg, f"diag_checkpoints_{a.agent}"), a.overwrite)
    scens = scenarios(cfg, d["range"], loads, n, "development"); models = Path(a.models_dir); seeds = a.seeds or d["legacy_seeds"]
    bl = build_baselines(); ref = {k: np.array([run_baseline(net, bl[k], sc)["accepted_offered_ratio"] for _, _, sc in scens]) for k in ("minhop", "unit_weight_dijkstra", "shortest_feasible_k", "shortest_feasible")}
    rows = []; eq = d["equivalence_margin"] * 100
    with RunManifest(out / "manifest.json", "x_02_diag_checkpoints", cfg, vars(a), seeds={"scenarios": f"{d['range']}[0:{n}) x loads {loads}"}, extra={"topology_sha256": sha, "status_of_results": "EXPLORATORY"}) as mf:
        for s in seeds:
            for ck in a.ckpts:
                m = loader(models / a.agent, s, ck); ratios, r = diagnose(net, a.agent, m, scens, bl["unit_weight_dijkstra"])
                row = dict(agent=a.agent, seed=s, ckpt=ck, mean_ratio=float(ratios.mean()), **r)
                for k, v in ref.items():
                    sd = summarize_diff((ratios - v)[None, :], B=a.boot, rng=np.random.default_rng(0)); row[f"diff_vs_{k}_pp"] = sd["mean_pp"]; row[f"diff_vs_{k}_ci_lo"] = sd["ci_scen_lo"]; row[f"diff_vs_{k}_ci_hi"] = sd["ci_scen_hi"]
                rows.append(row); log.info("%s seed %s %s ratio=%.4f", a.agent, s, ck, row["mean_ratio"])
        with open(out / "per_model.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        mf.add_output(out / "per_model.csv")
        best = [r for r in rows if r["ckpt"] == "best"] or rows
        summ = dict(agent=a.agent, n_models=len(best), reference_pooled_means={k: float(v.mean()) for k, v in ref.items()}, status="EXPLORATORY")
        if a.agent == "enhance":
            thr = d["b_route_equal_threshold"]
            summ.update(n_models_route_equal_unit_ge_threshold=int(sum(r["frac_route_equals_unit_dijkstra"] >= thr for r in best)), route_equal_threshold=thr,
                        n_models_ratio_equivalent_to_unit=int(sum(abs(r["diff_vs_unit_weight_dijkstra_ci_lo"]) <= eq and abs(r["diff_vs_unit_weight_dijkstra_ci_hi"]) <= eq for r in best)),
                        mean_frac_components_outside_bounds=float(np.mean([r["frac_components_outside_bounds"] for r in best])), mean_policy_std=float(np.mean([r["policy_std_mean"] for r in best])),
                        reading="If most models choose the unit-weight route (>= threshold) and are ratio-equivalent to unit-weight Dijkstra, trained Agent B is behaviourally a static min-hop router: report as a negative result and do not treat attacks on it as attacks on a learned routing policy.")
        else:
            summ.update(mean_ratio_best=float(np.mean([r["mean_ratio"] for r in best])), mean_frac_cand0=float(np.mean([r["frac_route_is_minhop_cand0"] for r in best])))
        atomic_write_text(out / "summary.json", json.dumps(summ, indent=2)); mf.add_output(out / "summary.json")
    return summ, rows


def main():
    ap = cli.add_common(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--agent", choices=["enhance", "replace"], required=True); ap.add_argument("--models-dir", type=Path, default=Path(os.environ.get("P2X_PROJECT_ROOT", ROOT)) / "results/models")
    ap.add_argument("--seeds", type=int, nargs="+", default=None); ap.add_argument("--ckpts", nargs="+", default=None, choices=["best", "final"])
    ap.add_argument("--n-per-load", type=int, default=None); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); cli.setup(a.verbose); cfg = cli.get_cfg(a.config); a.ckpts = a.ckpts or cfg["diagnostics"]["ckpts"]
    cli.need_dir(a.models_dir / a.agent, "models directory"); net, sha = cli.real_network(cfg)
    summ, _ = execute(cfg, a, net, sha); print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
