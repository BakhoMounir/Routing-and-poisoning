#!/usr/bin/env python3
"""Stage B - cheap READ-ONLY integrity check (~2-3 min CPU). Installs nothing, trains nothing.
 1. compares installed package versions with requirements.txt and with the versions recorded in each meta.json;
 2. loads every shipped checkpoint, recomputes final_param_sha256 (bit-level load fidelity) and re-evaluates best.zip on the
    LEGACY selection validation set, comparing with meta.json::best_val.
Decision: if a checkpoint does not reproduce its recorded validation score (|diff| > tolerance) the local torch/SB3 build is not
equivalent for that model; report it and evaluate shipped checkpoints only where they reproduce."""
import os
import argparse, csv, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from p2x import cli
from p2x.cli import log
from p2x.config import atomic_write_text, results_dir, scenarios
from p2x.manifest import RunManifest, env_versions


def main():
    ap = cli.add_common(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--models-dir", type=Path, default=Path(os.environ.get("P2X_PROJECT_ROOT", ROOT)) / "results/models")
    ap.add_argument("--agents", nargs="+", default=["replace", "enhance"], choices=["replace", "enhance"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--hash-only", action="store_true", help="skip the validation re-evaluation (seconds instead of minutes)")
    a = ap.parse_args(); cli.setup(a.verbose); cfg = cli.get_cfg(a.config)
    models = cli.need_dir(a.models_dir, "models directory")
    seeds = a.seeds or cfg["diagnostics"]["legacy_seeds"]
    out = cli.prepare_out(a.out or results_dir(cfg, "env_check"), a.overwrite)
    pins = dict(l.strip().split("==") for l in (ROOT / "requirements.txt").read_text().splitlines() if "==" in l and not l.startswith("#")) if (ROOT / "requirements.txt").is_file() else {}
    have = env_versions(); issues = []
    for k, want in pins.items():
        got = have.get(k) or have.get(k.replace("_", "-"))
        flag = "OK" if got and got.split("+")[0] == want else ("MISSING" if not got else "DIFFERENT")
        if flag != "OK": issues.append(f"{k}: installed={got} pinned={want}")
        log.info("%-20s installed=%-14s pinned=%-10s %s", k, got, want, flag)
    if have.get("torch") is None or have.get("stable-baselines3") is None:
        cli.die("torch / stable-baselines3 not importable in this interpreter. Activate the project venv; nothing was installed.")
    import numpy as np
    from p2x.rollout import evaluate_fast, load_model, param_sha256
    net, ned_sha = cli.real_network(cfg); rows = []
    val = None if a.hash_only else scenarios(cfg, "val_selection_v1", cfg["loads"], cfg["diagnostics"]["integrity_n_per_load"], "integrity")
    with RunManifest(out / "manifest.json", "x_01_check_env_and_checkpoints", cfg, vars(a), seeds={"integrity_val": "val_selection_v1"}) as mf:
        for ag in a.agents:
            for s in seeds:
                meta_p = models / ag / f"seed{s}" / "meta.json"
                if not meta_p.is_file(): log.warning("no meta.json for %s seed %s - skipped", ag, s); continue
                meta = json.loads(meta_p.read_text()); mf.add_input(meta_p)
                fin = load_model(models / ag, s, "final"); h_ok = param_sha256(fin) == meta["final_param_sha256"]
                dv = None; v = None
                if val is not None:
                    best = load_model(models / ag, s, "best"); v = float(np.mean(evaluate_fast(net, ag, best, [sc for _, _, sc in val]))); dv = v - meta["best_val"]
                tol = cfg["diagnostics"]["integrity_tolerance"]
                status = "REPRODUCED" if (h_ok and (dv is None or abs(dv) <= tol)) else ("DIVERGES" if (dv is not None and abs(dv) > 1e-3) else "FLOAT-LEVEL")
                rows.append(dict(agent=ag, seed=s, param_hash_match=h_ok, recorded_best_val=meta["best_val"], reevaluated_best_val=v, diff=dv,
                                 trained_with=f"torch {meta['versions']['torch']} numpy {meta['versions']['numpy']}", status=status))
                log.info("%s seed %s: hash_match=%s diff=%s -> %s", ag, s, h_ok, dv, status)
        if not rows: cli.die(f"no checkpoints with meta.json found under {models} for agents {a.agents}")
        mf.set("topology_sha256", ned_sha)
        with open(out / "checkpoint_integrity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        mf.add_output(out / "checkpoint_integrity.csv"); mf.set("version_issues", issues); mf.set("n_checkpoints", len(rows))
        mf.set("n_diverging", sum(r["status"] == "DIVERGES" for r in rows))
    print(f"\n{len(rows)} checkpoints checked; version issues: {issues or 'none'}; DIVERGES: {sum(r['status']=='DIVERGES' for r in rows)}; wrote {out}")


if __name__ == "__main__":
    main()
