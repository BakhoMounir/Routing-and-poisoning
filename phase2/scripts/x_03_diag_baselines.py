#!/usr/bin/env python3
"""Stage B/C - NO-TRAINING diagnostics on the development validation set (val_v2). Seconds to ~1 minute on one CPU core, no torch needed.
Questions (all EXPLORATORY: they steer decisions, they are not confirmatory evidence):
  Q1  Is Agent A's K=4 candidate restriction binding?  shortest_feasible vs shortest_feasible_k (same 4 candidates).
  Q2  How often is the unrestricted shortest-feasible path inside the 4 candidates?
  Q3  Can a transparent utilization heuristic work through Agent B's Dijkstra pipeline?  grid vs the unit-weight-Dijkstra null.
SELECTION-BIAS CONTROL: the heuristic parameters are chosen on the SELECTION half of val_v2 (scenario idx < n//2); every decision statistic about
the chosen heuristic is computed on the disjoint ESTIMATION half (idx >= n//2). Full-set numbers are printed as 'optimistic'. The only
confirmatory test of the heuristic is P4 on the fresh confirmatory set (x_06). The heuristic is NOT a trained model."""
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from p2.baselines import shortest_feasible
from p2.sim import Episode
from p2x import cli
from p2x.cli import log
from p2x.config import atomic_write_text, results_dir, scenarios
from p2x.manifest import RunManifest
from p2x.policies import build_baselines, heuristic_grid
from p2x.rollout import collect_rows, run_baseline
from p2x.stats import summarize_diff


def hid(h): return f"util_w_p{h['p']:g}_m{h['infeasible_multiplier']:g}"


def classify(s, eq_pp):
    """Three-way reading of a CI (pp): 'positive' (lo > eq), 'negative' (hi < -eq), 'equivalent' (CI inside +-eq), else 'inconclusive'."""
    lo, hi = s["ci_scen_lo"], s["ci_scen_hi"]
    return "positive" if lo > eq_pp else "negative" if hi < -eq_pp else "equivalent" if (lo >= -eq_pp and hi <= eq_pp) else "inconclusive"


def execute(cfg, a, net, ned_sha):
    d = cfg["diagnostics"]; n = a.n_per_load or d["n_per_load"]; loads = cfg["loads"]
    if n < 4: cli.die("--n-per-load must be >= 4 (needs a selection and an estimation half)")
    out = cli.prepare_out(a.out or results_dir(cfg, "diag_baselines"), a.overwrite)
    scens = scenarios(cfg, d["range"], loads, n, "development")
    grid = heuristic_grid(cfg); methods = dict(build_baselines())
    for h in grid: methods[hid(h)] = build_baselines(h)["util_weight_dijkstra"]
    half = n // 2; sel_idx, est_idx = np.arange(0, half), np.arange(half, n)
    rows = []
    with RunManifest(out / "manifest.json", "x_03_diag_baselines", cfg, vars(a), seeds={"scenarios": f"{d['range']}[0:{n}) x loads {loads}", "selection_half": f"idx<{half}", "estimation_half": f"idx>={half}"},
                     extra={"topology_sha256": ned_sha, "status_of_results": "EXPLORATORY (development set)"}) as mf:
        for name, fn in methods.items():
            rows += collect_rows(name, -1, scens, lambda sc, fn=fn: run_baseline(net, fn, sc)); log.info("ran %-22s (%d episodes)", name, len(scens))
        M = {m: np.array([[r["accepted_offered_ratio"] for r in rows if r["method"] == m and r["load"] == L] for L in loads]) for m in methods}   # (loads, n)
        rng = np.random.default_rng(cfg["confirmatory"]["bootstrap_seed"])
        def diff(t, r, idx):
            o = {str(L): summarize_diff((M[t][li, idx] - M[r][li, idx])[None, :], B=a.boot, rng=rng) for li, L in enumerate(loads)}
            o["pooled"] = summarize_diff((M[t][:, idx].mean(0) - M[r][:, idx].mean(0))[None, :], B=a.boot, rng=rng); return o
        sel_mean = {hid(h): float(M[hid(h)][:, sel_idx].mean()) for h in grid}
        sel = max(grid, key=lambda h: (round(sel_mean[hid(h)], 12), -h["p"], -h["infeasible_multiplier"])); sid = hid(sel)
        full_mean = {m: float(M[m].mean()) for m in methods}
        allidx = np.arange(n)
        D_expl = {f"{t} - {r}": diff(t, r, allidx) for t, r in [("shortest_feasible_k", "minhop"), ("unit_weight_dijkstra", "minhop"), ("shortest_feasible", "shortest_feasible_k"),
                                                                 ("greedy_widest", "shortest_feasible_k"), ("shortest_feasible", "minhop")]}
        D_heldout = {f"{sid} - unit_weight_dijkstra": diff(sid, "unit_weight_dijkstra", est_idx), f"{sid} - shortest_feasible": diff(sid, "shortest_feasible", est_idx)}
        D_optimistic = {f"{hid(h)} - unit_weight_dijkstra": diff(hid(h), "unit_weight_dijkstra", allidx)["pooled"] for h in grid}
        cov = {str(L): [] for L in loads}; extra = {str(L): [] for L in loads}
        for _, L, sc in scens:
            ep = Episode(net, sc); inK = 0; ex = 0
            while not ep.done:
                s, dd, _ = ep.demand(); e = shortest_feasible(net, ep, ep.true_util())
                inK += int(any(np.array_equal(e, c) for _, c in net.candidates(s, dd))); ex += len(e) - len(net.candidates(s, dd)[0][1]); ep.route(e)
            cov[str(L)].append(inK / len(sc.vol)); extra[str(L)].append(ex / len(sc.vol))
        coverage = {L: dict(frac_decisions_with_sf_path_in_K=float(np.mean(v)), mean_extra_hops_vs_minhop=float(np.mean(extra[L]))) for L, v in cov.items()}
        eq, gain = d["equivalence_margin"] * 100, d["meaningful_gain"] * 100
        k = D_expl["shortest_feasible - shortest_feasible_k"]["pooled"]; hu = D_heldout[f"{sid} - unit_weight_dijkstra"]["pooled"]; hs = D_heldout[f"{sid} - shortest_feasible"]["pooled"]
        dec = dict(status="EXPLORATORY - steers decisions only; confirmatory tests are P1-P4 on the fresh set",
                   K_restriction_cost_pp=dict(mean=k["mean_pp"], ci=[k["ci_scen_lo"], k["ci_scen_hi"]], reading=classify(k, eq), meaning="shortest_feasible minus shortest_feasible_k; 'positive' = restriction binding, 'equivalent' = negligible, 'inconclusive' = cannot tell"),
                   heuristic_minus_unit_pp_HELDOUT=dict(mean=hu["mean_pp"], ci=[hu["ci_scen_lo"], hu["ci_scen_hi"]], reading=classify(hu, eq)),
                   heuristic_minus_shortest_feasible_pp_HELDOUT=dict(mean=hs["mean_pp"], ci=[hs["ci_scen_lo"], hs["ci_scen_hi"]], non_inferior_within_meaningful_gain=bool(hs["ci_scen_lo"] >= -gain)),
                   thresholds_pp=dict(equivalence=eq, meaningful=gain), scenario_only_ci="CIs here reflect scenario sampling only (no trained models involved)")
        selection = dict(selected=sel, selection_rule="max mean accepted_offered_ratio on the SELECTION half (idx<n//2) of val_v2; ties -> smaller p, then smaller multiplier",
                         n_per_load=n, selection_half_means=sel_mean, full_set_means_optimistic={m: full_mean[m] for m in full_mean})
        flat = ["method", "model_seed", "scenario_idx", "scenario_seed", "load", "accepted_offered_ratio", "blocked_volume_ratio", "max_utilization", "mean_hops_accepted", "frac_route_is_minhop_cand0", "replay_ok"]
        with open(out / "episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flat, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        mf.add_output(out / "episodes.csv")
        for fn, obj in (("paired_diffs_exploratory_pp.json", D_expl), ("paired_diffs_heldout_pp.json", D_heldout), ("grid_vs_unit_OPTIMISTIC_pp.json", D_optimistic),
                        ("coverage.json", coverage), ("heuristic_selection.json", selection), ("decisions.json", dec)):
            atomic_write_text(out / fn, json.dumps(obj, indent=2, default=float)); mf.add_output(out / fn)
    return dict(full_mean=full_mean, coverage=coverage, selected=sel, decisions=dec, out=out)


def main():
    ap = cli.add_common(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--n-per-load", type=int, default=None); ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args(); cli.setup(a.verbose); cfg = cli.get_cfg(a.config); net, sha = cli.real_network(cfg)
    r = execute(cfg, a, net, sha)
    print("\nmean accepted/offered ratio on val_v2, all loads equal weight (OPTIMISTIC for the grid: selection and evaluation overlap):")
    for m in sorted(r["full_mean"], key=r["full_mean"].get, reverse=True): print(f"  {m:<24}{r['full_mean'][m]:.4f}")
    print("\ncoverage:", json.dumps(r["coverage"])); print("selected heuristic:", r["selected"]); print(json.dumps(r["decisions"], indent=1)); print(f"\noutputs in {r['out']}")


if __name__ == "__main__":
    main()
