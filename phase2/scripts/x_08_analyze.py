#!/usr/bin/env python3
"""Analysis of confirmatory / poison episode tables (pandas+numpy; no torch). Effects in PERCENTAGE POINTS of accepted_offered_ratio.
Primary family (config confirmatory.primary_family; per load): Holm-adjusted bootstrap p-values and Bonferroni-adjusted crossed-bootstrap CIs.
Three interval types are reported separately: scenario-only (given the trained models), across-model-seed (t), and crossed (both).
Secondary comparisons are exploratory. Use --exclude-model-seeds for the Agent B environment-drift sensitivity run (seed 42 trained under a different torch build)."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from p2x import analysis, cli
from p2x.cli import log
from p2x.config import results_dir


def main():
    ap = cli.add_common(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--stage", choices=["confirmatory", "poison"], required=True); ap.add_argument("--run-dir", type=Path, default=None, help="directory containing episodes.csv")
    ap.add_argument("--preset", choices=["smoke", "pilot", "full"], default="full"); ap.add_argument("--exclude-model-seeds", type=int, nargs="*", default=[])
    a = ap.parse_args(); cli.setup(a.verbose); cfg = cli.get_cfg(a.config)
    import pandas as pd
    run = a.run_dir or (results_dir(cfg, cfg["confirmatory"]["run_id"]) if a.stage == "confirmatory" else results_dir(cfg, f"poison_{a.preset}"))
    ep = cli.need_file(run / "episodes.csv", "episodes.csv (run the evaluation first)"); df = pd.read_csv(ep)
    for c in ("replay_ok", "caps_ok"):
        if c in df and not df[c].astype(str).eq("True").all(): cli.die(f"invariant column {c} is not all True - results invalid")
    out = cli.prepare_out(a.out or run / ("analysis" + ("_excl" + "-".join(map(str, a.exclude_model_seeds)) if a.exclude_model_seeds else "")), a.overwrite)
    c = cfg["confirmatory"]; rng = np.random.default_rng(c["bootstrap_seed"]); ex = tuple(a.exclude_model_seeds); B = c["bootstrap_B"]
    if a.stage == "confirmatory":
        loads = sorted(df.load.unique()); analysis.summary_table(df, loads, ex).to_csv(out / "summary_by_method_load.csv", index=False)
        prim = analysis.compare(df, [(x["id"], x["test"], x["ref"]) for x in c["primary_family"]], loads, c["alpha"], B, rng, ex, "primary")
        sec = analysis.compare(df, [(f"S{i+1}", x["test"], x["ref"]) for i, x in enumerate(c["secondary"])], loads, c["alpha"], B, rng, ex, "secondary_exploratory")
        prim.to_csv(out / "primary_comparisons_pp.csv", index=False); sec.to_csv(out / "secondary_comparisons_pp_EXPLORATORY.csv", index=False)
        pd.set_option("display.width", 250); cols = ["id", "test", "ref", "load", "mean_pp", "ci_scen_lo", "ci_scen_hi", "ci_cross_bonf_lo", "ci_cross_bonf_hi", "p_holm", "reject_holm_alpha"]
        print(prim[[x for x in cols if x in prim]].round(3).to_string(index=False))
        miss = prim[~prim.computed]; log.info("primary tests not computable (method absent): %d", len(miss))
    else:
        loads = sorted(df.load.unique()); P = cfg["poisoning"]; eps = sorted(df.eps.unique()); eps = [e for e in eps if e > 0]
        t = analysis.poison_tables(df, sorted(df.method.unique()), P["attacks"], eps, loads, P["primary"], c["alpha"], B, rng, ex)
        t.to_csv(out / "poison_effects_pp.csv", index=False)
        flips = df[df.attack != "clean"].groupby(["method", "attack"]).frac_decisions_flipped.mean().reset_index(); flips.to_csv(out / "decision_flip_rates.csv", index=False)
        print(t[t.primary].round(3).to_string(index=False) if len(t) and t.primary.any() else "no primary cells present in this preset")
        log.info("claim policy: 'attack success' requires the crossed CI of (attacked - clean) to exclude 0 AND the contrast vs random_sign to exclude 0 (Holm over the primary cells); everything else is exploratory")
    log.info("wrote tables to %s", out)


if __name__ == "__main__":
    main()
