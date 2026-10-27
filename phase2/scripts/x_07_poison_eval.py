#!/usr/bin/env python3
"""Stage F - TEST-TIME TELEMETRY POISONING (prepared; DRY RUN by default, --run to execute). Threat model: docs/THREAT_MODEL.md.
Perturbs ONLY the utilization vector observed by a policy: obs = clip(true + delta, 0, 1), |delta_e| <= eps. True loads, capacities, traffic, reward and
model parameters are untouched; every episode is replay-verified and model parameter hashes are compared before/after.
Gates: presets 'pilot'/'full' need a valid freeze AND a completed confirmatory run (attack matrix is predeclared in the frozen config; no tuning).
Preset 'smoke' (3 scenarios, development set) only validates the pipeline; its numbers must not be interpreted or used to change the matrix.
Clean controls (attack 'clean') and the random_sign noise control at equal eps are always part of the matrix."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from p2x import cli
from p2x.cli import log
from p2x.config import agent_models_dir, freeze_path, results_dir, scenarios
from p2x.manifest import RunManifest
from p2x.parts import merge_parts, part_name, verify_freeze, write_part
from p2x.policies import build_baselines
from p2x.rollout import InvariantViolation, load_model, param_sha256, run_baseline, run_model
from p2x.telemetry import make_telemetry

COST_S = {"model_replace": 0.08, "model_enhance": 0.04, "baseline": 0.01}      # per episode incl. counterfactual pass; extrapolated from legacy eval_clean.log, indicative only


def matrix_of(cfg, preset):
    p = cfg["poisoning"]["presets"][preset]
    cfgs = [("clean", 0.0)] + [(k, e) for k in cfg["poisoning"]["attacks"] for e in p["eps"]]
    return p, cfgs


def plan(cfg, preset, models_present, base_targets=None):
    p, cfgs = matrix_of(cfg, preset); n_ep = len(p["loads"]) * p["n_per_load"] * len(cfgs)
    nb = len(base_targets if base_targets is not None else cfg["poisoning"]["targets_baselines"]); est = nb * n_ep * COST_S["baseline"]
    for aid, seeds in models_present.items():
        g = next(a["agent"] for a in cfg["confirmatory"]["agents"] if a["id"] == aid); est += len(seeds) * n_ep * COST_S[f"model_{g}"]
    return dict(preset=preset, attack_configs=len(cfgs), episodes_per_target=n_ep, baseline_targets=nb, model_targets={k: len(v) for k, v in models_present.items()}, est_cpu_minutes=round(est / 60, 1))


def execute(cfg, a, net, sha, loader=load_model, hash_fn=param_sha256):
    P = cfg["poisoning"]; p, cfgs = matrix_of(cfg, a.preset); smoke = a.preset == "smoke"; k = P["k_edges"]
    ids = {x["id"]: x for x in cfg["confirmatory"]["agents"]}
    if smoke:
        heur_p = results_dir(cfg, "diag_baselines", "heuristic_selection.json")
        heur = json.loads(heur_p.read_text())["selected"] if heur_p.is_file() else dict(p=1.0, floor=0.1, infeasible_multiplier=1.0)
        present = {t: [s for s in p["model_seeds"] if (agent_models_dir(ids[t]) / f"seed{s}" / f"{ids[t]['ckpt']}.zip").is_file()] for t in P["targets_models"]}
    else:
        try:
            fr = verify_freeze(cfg, freeze_path(cfg), sha, ROOT)
        except ValueError as e:
            cli.die(f"freeze check failed: {e}")
        cm = results_dir(cfg, cfg["confirmatory"]["run_id"], "manifest.json")
        if not cm.is_file() or json.loads(cm.read_text()).get("status") != "completed":
            cli.die("confirmatory evaluation has not completed; poisoning is gated on a finished clean stage")
        heur = fr["heuristic"]; present = {t: [s for s in p["model_seeds"] if str(s) in fr["models"].get(t, {})] for t in P["targets_models"]}
    want = getattr(a, "targets", None)
    present = {t: v for t, v in present.items() if v and not a.baselines_only and (not want or t in want)}
    base_targets = [t for t in P["targets_baselines"] if not want or t in want]
    pl = plan(cfg, a.preset, present, base_targets); log.info("PLAN %s", json.dumps(pl))
    if not a.run:
        log.info("DRY RUN - nothing executed. Add --run."); return None
    out = a.out or results_dir(cfg, f"poison_{a.preset}"); parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    scens = scenarios(cfg, p["range"], p["loads"], p["n_per_load"], p["role"]); bl = build_baselines(heur)
    def episode_rows(target, seed, runner):
        rows = []
        for kind, eps in cfgs:
            for i, L, sc in scens:
                tele = make_telemetry(kind, eps, k, sc.seed, L)
                m = runner(sc, tele, kind != "clean")
                if m["tele_max_abs_delta"] > eps + 1e-9: raise InvariantViolation(f"perturbation {m['tele_max_abs_delta']} exceeds budget {eps}")
                rows.append(dict(method=target, model_seed=seed, attack=("clean" if kind == "clean" else f"{kind}@{eps:g}"), kind=kind, eps=eps, k_edges=k,
                                 scenario_idx=i, scenario_seed=sc.seed, load=L, **m))
        return rows
    with RunManifest(out / "manifest.json", f"x_07_poison_eval[{a.preset}]", cfg, vars(a), seeds={"scenarios": f"{p['range']}[0:{p['n_per_load']}) x loads {p['loads']}"},
                     extra={"topology_sha256": sha, "plan": pl, "heuristic": heur, "status_of_results": "pipeline check only" if smoke else "pre-declared Stage F run"}) as mf:
        for t in base_targets:
            pf = parts / part_name(t, -1)
            if not pf.is_file(): write_part(pf, episode_rows(t, -1, lambda sc, tele, cf, f=bl[t]: run_baseline(net, f, sc, telemetry=tele, counterfactual=cf))); log.info("done baseline %s", t)
        for t, seeds in present.items():
            for s in seeds:
                pf = parts / part_name(t, s)
                if pf.is_file(): continue
                m = loader(agent_models_dir(ids[t]), s, ids[t]["ckpt"]); h0 = hash_fn(m); g = ids[t]["agent"]
                rows = episode_rows(t, s, lambda sc, tele, cf: run_model(net, g, m, sc, telemetry=tele, counterfactual=cf))
                if hash_fn(m) != h0: raise InvariantViolation(f"model parameters changed during poisoned evaluation: {t} seed {s}")
                for r in rows: r["param_sha_prefix"] = h0[:12]; r["param_unchanged"] = True
                write_part(pf, rows); log.info("done %s seed %s", t, s)
        n = merge_parts(parts, out / "episodes.csv"); mf.add_output(out / "episodes.csv"); mf.set("n_rows", n)
        import csv
        with open(out / "episodes.csv", newline="") as f:
            R = list(csv.DictReader(f))
        mf.set("invariants", dict(all_replay_ok=all(r["replay_ok"] == "True" for r in R), all_caps_ok=all(r["caps_ok"] == "True" for r in R),
                                  max_delta_over_budget=max(float(r["tele_max_abs_delta"]) - float(r["eps"]) for r in R)))
    return out


def main():
    ap = cli.add_common(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--preset", choices=["smoke", "pilot", "full"], default="smoke"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--baselines-only", action="store_true", help="skip trained models (no torch needed)")
    ap.add_argument("--targets", nargs="+", default=None, help="restrict to these target ids (split a long run across processes; parts are resumable)")
    a = ap.parse_args(); cli.setup(a.verbose); cfg = cli.get_cfg(a.config); net, sha = cli.real_network(cfg)
    if a.run and not a.baselines_only:
        try:
            import stable_baselines3  # noqa
        except ImportError:
            cli.die("torch / stable-baselines3 not importable; activate the project venv (or use --baselines-only).")
    execute(cfg, a, net, sha)


if __name__ == "__main__":
    main()
