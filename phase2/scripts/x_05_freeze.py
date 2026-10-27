#!/usr/bin/env python3
"""Stage D->E gate. Records, BEFORE any confirmatory evaluation, exactly what will be tested: config hash, core-code hash, topology hash,
checkpoint hashes (per agent id / seed), and the selected heuristic. x_06 refuses to run without a matching freeze; after the confirmatory
LOCK exists the freeze can no longer be replaced. Read-only on models. Requires --yes."""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from p2x import cli
from p2x.cli import log
from p2x.config import agent_models_dir, atomic_write_text, freeze_path, lock_path, results_dir, sha256_file
from p2x.manifest import core_code_sha256, env_versions


def build_freeze(cfg, heur_path, ned_sha):
    models, notes = {}, []
    for ag in cfg["confirmatory"]["agents"]:
        d = agent_models_dir(ag); entry = {}
        for s in ag["seeds"]:
            z = d / f"seed{s}" / f"{ag['ckpt']}.zip"; mp = d / f"seed{s}" / "meta.json"
            if z.is_file() and mp.is_file():
                meta = json.loads(mp.read_text()); entry[str(s)] = dict(path=(str(z.relative_to(ROOT)) if ROOT in z.parents else str(z)), sha256=sha256_file(z), best_val=meta.get("best_val"),
                                                                       stop_reason=meta.get("stop_reason"), torch=meta["versions"].get("torch"))
        if len(entry) != len(ag["seeds"]):
            if ag["optional"]:
                notes.append(f"{ag['id']}: {len(entry)}/{len(ag['seeds'])} seeds present -> treated as ABSENT (primary comparisons P1/P2 will be not-computed)"); entry = {}
            else:
                raise SystemExit(f"required agent {ag['id']} incomplete: {len(entry)}/{len(ag['seeds'])} checkpoints with meta.json")
        models[ag["id"]] = entry
    heur = json.loads(Path(heur_path).read_text())["selected"]
    torch_groups = sorted({v["torch"] for e in models.values() for v in e.values()})
    if len(torch_groups) > 1: notes.append(f"checkpoints were trained under several torch builds: {torch_groups} (documented deviation)")
    return dict(created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), config_sha256=cfg["_sha256"], core_code_sha256=core_code_sha256(),
                topology_sha256=ned_sha, models=models, heuristic=heur, heuristic_source=str(heur_path), environment=env_versions(), notes=notes)


def main():
    ap = cli.add_common(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--heuristic", type=Path, default=None, help="heuristic_selection.json from x_03 (default: <results_root>/diag_baselines/heuristic_selection.json)")
    ap.add_argument("--yes", action="store_true", help="confirm that methodology, hyper-parameters and attack matrix are final")
    a = ap.parse_args(); cli.setup(a.verbose); cfg = cli.get_cfg(a.config); _, sha = cli.real_network(cfg)
    fp = freeze_path(cfg); hp = a.heuristic or results_dir(cfg, "diag_baselines", "heuristic_selection.json")
    cli.need_file(hp, "heuristic selection (run scripts/x_03_diag_baselines.py first)")
    if lock_path(cfg, cfg["confirmatory"]["run_id"]).is_file(): cli.die("confirmatory LOCK exists: the freeze can no longer be changed")
    if fp.is_file() and not a.overwrite: cli.die(f"{fp} exists; pass --overwrite only if NO confirmatory data has been looked at")
    fr = build_freeze(cfg, hp, sha)
    log.info("freeze summary: agents=%s heuristic=%s notes=%s", {k: len(v) for k, v in fr["models"].items()}, fr["heuristic"], fr["notes"])
    if not a.yes: log.info("DRY RUN - freeze not written. Re-run with --yes when the checklist in docs/02_EXPERIMENT_PROTOCOL.md is satisfied."); return
    atomic_write_text(fp, json.dumps(fr, indent=2)); log.info("wrote %s", fp)


if __name__ == "__main__":
    main()
