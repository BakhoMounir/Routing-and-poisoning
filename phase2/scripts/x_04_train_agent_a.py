#!/usr/bin/env python3
"""Stage D - OPTIONAL, EXPENSIVE: longer Agent A training (budget/stopping rule in configs/x_experiments.json::train_agent_a).
DEFAULT IS A DRY RUN: prints the plan and cost estimate and trains nothing. Pass --run to train.
Writes ONLY to <results_root>/models/<run_id>/seed<N>/ ; completed seeds (meta.json present) are skipped; interrupted seeds resume from last.zip
(statistically, not bit-wise, equivalent). Legacy checkpoints are never touched. Does not train Agent B.
Decision it informs: does Agent A keep improving past 150k steps (and does the gain survive the stopping rule)? Not a convergence claim."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from p2x import cli
from p2x.cli import log
from p2x.config import assert_not_protected, results_dir, scenarios


def main():
    ap = cli.add_common(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--preset", choices=["pilot", "full"], default="pilot", help="pilot = config pilot_seeds (3), full = 10 seeds")
    g.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--run", action="store_true", help="actually train (otherwise dry run)")
    a = ap.parse_args(); cli.setup(a.verbose); cfg = cli.get_cfg(a.config); t = cfg["train_agent_a"]
    seeds = a.seeds or t[f"{a.preset}_seeds"]; root = a.out or results_dir(cfg, "models", t["run_id"]); assert_not_protected(cfg, root)
    lo, hi = t["est_wall_s_per_seed_range"]
    log.info("PLAN: agent=replace run_id=%s seeds=%s max_steps=%d eval_every=%d (<=%d evals) val=%s %d/load", t["run_id"], seeds, t["max_steps"], t["eval_every"], t["max_steps"] // t["eval_every"], t["val_range"], t["n_val_per_load"])
    log.info("stopping rule: %s", t["stopping"])
    log.info("ESTIMATED cost (extrapolated from legacy logs: 150k steps took ~195-320 s on 1 sandbox core; YOUR machine will differ): %.0f-%.0f min total, 1 core per seed (seeds can run in parallel processes)", len(seeds) * lo / 60, len(seeds) * hi / 60)
    log.info("output: %s", root)
    if not a.run:
        log.info("DRY RUN - nothing trained. Re-run with --run to start."); return
    net, sha = cli.real_network(cfg)
    try:
        import torch, stable_baselines3  # noqa
    except ImportError:
        cli.die("torch / stable-baselines3 not importable; activate the project venv. Nothing installed automatically.")
    from p2x.train_long import train_agent_a
    val = scenarios(cfg, t["val_range"], cfg["loads"], t["n_val_per_load"], "development")
    for s in seeds:
        train_agent_a(cfg, s, root / f"seed{s}", val, log=log.info)
    log.info("finished seeds %s", seeds)


if __name__ == "__main__":
    main()
