#!/usr/bin/env python3
"""Stage E - CONFIRMATORY clean evaluation on a FRESH, independently seeded test set (range test_confirmatory_v2).
Requires a valid freeze (x_05). Creates <results_root>/<run_id>/LOCK.json on first use: the range is then consumed; re-running only RESUMES missing
parts for the same freeze. NO tuning is allowed after this point. `--smoke` runs 5 scenarios/load on the DEVELOPMENT set instead (pipeline check,
no freeze/lock, output in <run_id>_smoke, uses the unfrozen config models that exist). Output metric: accepted_offered_ratio (volume, not packets)."""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from p2x import cli
from p2x.cli import log
from p2x.config import agent_models_dir, atomic_write_text, lock_path, freeze_path, results_dir, scenarios
from p2x.manifest import RunManifest
from p2x.parts import merge_parts, part_name, verify_freeze, write_part
from p2x.policies import build_baselines
from p2x.rollout import collect_rows, load_model, run_baseline, run_model


def execute(cfg, a, net, sha, loader=load_model):
    c = cfg["confirmatory"]; loads = cfg["loads"]
    if a.smoke:
        rng_name, role, n, run = cfg["diagnostics"]["range"], "development", 5, c["run_id"] + "_smoke"
    else:
        rng_name, role, n, run = c["range"], "confirmatory", c["n_per_load"], c["run_id"]
    out = results_dir(cfg, run) if a.out is None else a.out; parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        hp = Path(a.heuristic) if a.heuristic else results_dir(cfg, "diag_baselines", "heuristic_selection.json")
        heur = json.loads(hp.read_text())["selected"] if hp.is_file() else dict(p=1.0, floor=0.1, infeasible_multiplier=1.0)
        avail = {ag["id"]: [s for s in ag["seeds"] if (agent_models_dir(ag) / f"seed{s}" / f"{ag['ckpt']}.zip").is_file()] for ag in c["agents"]}
        avail = {k: v[:2] for k, v in avail.items() if v}                                  # smoke: 2 seeds per available agent
    else:
        try:
            fr = verify_freeze(cfg, freeze_path(cfg), sha, ROOT)
        except ValueError as e:
            cli.die(f"freeze check failed: {e}")
        lp = lock_path(cfg, run); resumed = lp.is_file()
        if lp.is_file():
            if json.loads(lp.read_text())["freeze_config_sha256"] != fr["config_sha256"]: cli.die("LOCK belongs to a different freeze")
            log.warning("LOCK exists: RESUMING missing parts only (no new test data is consumed)")
        else:
            atomic_write_text(lp, json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), freeze_config_sha256=fr["config_sha256"], range=rng_name, n_per_load=n)))
        heur = fr["heuristic"]; avail = {k: sorted(int(s) for s in v) for k, v in fr["models"].items() if v}
    scens = scenarios(cfg, rng_name, loads, n, role); bl = build_baselines(heur); agents = {ag["id"]: ag for ag in c["agents"]}
    with RunManifest(out / "manifest.json", "x_06_confirmatory_eval" + ("_smoke" if a.smoke else ""), cfg, vars(a), seeds={"scenarios": f"{rng_name}[0:{n}) x loads {loads}"},
                     extra={"topology_sha256": sha, "heuristic": heur, "agents_present": {k: v for k, v in avail.items()}, "smoke": a.smoke, "resumed_existing_lock": (False if a.smoke else resumed)}) as mf:
        for name in c["baselines"]:
            pf = parts / part_name(name, -1)
            if not pf.is_file():
                write_part(pf, collect_rows(name, -1, scens, lambda sc, f=bl[name]: run_baseline(net, f, sc))); log.info("done %s", name)
        for aid, seeds in avail.items():
            ag = agents[aid]
            for s in seeds:
                pf = parts / part_name(aid, s)
                if pf.is_file(): continue
                m = loader(agent_models_dir(ag), s, ag["ckpt"])
                write_part(pf, collect_rows(aid, s, scens, lambda sc, m=m, g=ag["agent"]: run_model(net, g, m, sc))); log.info("done %s seed %s", aid, s)
        nrows = merge_parts(parts, out / "episodes.csv"); mf.add_output(out / "episodes.csv"); mf.set("n_rows", nrows)
    log.info("merged %d rows -> %s", nrows, out / "episodes.csv"); return out


def main():
    ap = cli.add_common(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--heuristic", type=Path, default=None)
    a = ap.parse_args(); cli.setup(a.verbose); cfg = cli.get_cfg(a.config); net, sha = cli.real_network(cfg)
    try:
        import stable_baselines3  # noqa
    except ImportError:
        cli.die("torch / stable-baselines3 not importable; activate the project venv.")
    execute(cfg, a, net, sha)


if __name__ == "__main__":
    main()
