#!/usr/bin/env python3
"""Stage A - STATIC, READ-ONLY audit of the existing Phase 2 project. Stdlib only: no torch/numpy needed, nothing is trained or imported from p2.
Verifies, from the files on disk, the findings listed in docs/01_METHODOLOGY_AUDIT.md and writes results/v2/static_audit/static_audit.json.
Exit code 0 always unless --strict is given (then 1 if any CONFIRMED problem)."""
import argparse, ast, csv, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CHECKS = []


def add(cid, sev, status, title, evidence):
    CHECKS.append(dict(id=cid, severity=sev, status=status, title=title, evidence=evidence))


def read(root, rel):
    p = root / rel
    return p.read_text(errors="replace") if p.is_file() else None


def read_csv(root, rel):
    p = root / rel
    if not p.is_file():
        return None
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def module_consts(src):
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    return out


def hp_from_train(src):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "HP" and isinstance(n.value, ast.Call):
            return {k.arg: ast.literal_eval(k.value) for k in n.value.keywords}
    return None


def main(a):
    root = Path(a.root).resolve()
    # ---- C01 tests
    t = read(root, "tests/test_phase2.py"); b = read(root, "src/p2/baselines.py")
    if t and b:
        tree = ast.parse(t); nb = 0
        for n in ast.parse(b).body:
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BASELINES" and isinstance(n.value, ast.Dict):
                nb = len(n.value.keys)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        n_expected = sum((nb if "parametrize" in ast.unparse(f) and "BASELINES" in ast.unparse(f) else 1) for f in funcs)
        logtxt = read(root, "results/pytest_phase2_handover.log") or ""
        m = re.search(r"(\d+) passed", logtxt)
        gaps = [k for k in ("eval_clean", "analyze_clean", "train_ppo", "PPO.load", "holm", "bootstrap") if k not in t]
        add("C01", "medium", "OK" if m and int(m.group(1)) == n_expected else "CHECK",
            "Test suite size matches the reported 23 passing tests",
            f"{len(funcs)} test functions, {n_expected} expanded cases; log reports '{m.group(0) if m else 'n/a'}'. "
            f"Not covered by any test (name absent from tests/test_phase2.py): {gaps}. Logs are from the original sandbox; re-run locally.")
    # ---- C02/C03 stale status documents vs artifacts
    st = read(root, "STATUS_AND_RESUME.md"); ni = read(root, "NOT_INCLUDED.md")
    meta_en = sorted((root / "results/models/enhance").glob("seed*/meta.json")) if (root / "results/models/enhance").is_dir() else []
    ep = (root / "results/clean/episodes_clean.csv").is_file(); figs = list((root / "figures").glob("*.png")) if (root / "figures").is_dir() else []
    done = (root / "results/logs/TRAINING_DONE").is_file(); reports = list((root / "reports").glob("*.md")) if (root / "reports").is_dir() else []
    if st:
        claims = [c for c in ("UNFINISHED", "NOT run", "NOT written", "not started", "NOT** run") if c in st]
        stale = bool(claims) and len(meta_en) == 10
        add("C02", "high", "CONFIRMED" if stale else "OK", "STATUS_AND_RESUME.md contradicts the artifacts on disk",
            f"STATUS says {claims}; disk has {len(meta_en)}/10 enhance meta.json, episodes_clean.csv={ep}, TRAINING_DONE={done}, figures={[f.name for f in figs]}, reports/*.md={[r.name for r in reports]}. Treat STATUS as superseded by docs/03_CURRENT_STATUS.md.")
    if ni:
        stale = "figures/` is empty" in ni or "figures/ is empty" in ni
        add("C03", "low", "CONFIRMED" if stale and figs else "OK", "NOT_INCLUDED.md says figures/tables do not exist",
            f"claims empty figures/ and no clean tables; disk has {len(figs)} figures and episodes_clean.csv={ep}")
    # ---- C04 dangling README references
    rd = read(root, "README.md")
    if rd:
        refs = sorted(set(re.findall(r"`([\w./-]+\.(?:md|py|json|csv|txt|sh))`", rd)))
        missing = [r for r in refs if "*" not in r and not (root / r).exists() and not (("/" not in r) and any(root.rglob(r)))]
        add("C04", "medium", "CONFIRMED" if missing else "OK", "README references files that are not present", f"missing: {missing}")
    # ---- C05/C06 config vs code vs meta
    cfg = json.loads(read(root, "configs/phase2_config.json") or "{}"); sim = module_consts(read(root, "src/p2/sim.py") or "")
    tp = read(root, "scripts/train_ppo.py"); hp = hp_from_train(tp) if tp else None
    x = json.loads(read(root, "configs/x_experiments.json") or "{}")
    probs = []
    if cfg and sim:
        for ck, sk in (("n_flows_per_episode", "N_FLOWS"), ("K_candidate_paths", "K_PATHS")):
            if cfg.get(ck) != sim.get(sk):
                probs.append(f"{ck}={cfg.get(ck)} vs sim.{sk}={sim.get(sk)}")
    if hp and x.get("ppo_hp"):
        xh = {k: v for k, v in x["ppo_hp"].items() if not k.startswith("_")}
        if xh != hp:
            probs.append(f"x_experiments.ppo_hp != train_ppo.HP: {xh} vs {hp}")
    metas = {ag: [json.loads(read(root, f"results/models/{ag}/seed{s}/meta.json") or "null") for s in range(42, 52)] for ag in ("replace", "enhance")}
    for ag, ms in metas.items():
        ms = [m for m in ms if m]
        if len({json.dumps(m["hyperparameters"], sort_keys=True) for m in ms}) > 1 or (hp and any(m["hyperparameters"] != hp for m in ms)):
            probs.append(f"{ag}: meta.json hyper-parameters differ across seeds or from train_ppo.HP")
        if len({m["total_timesteps"] for m in ms}) > 1:
            probs.append(f"{ag}: unequal training budgets {sorted({m['total_timesteps'] for m in ms})}")
    groups = {}
    for ag, ms in metas.items():
        for sd, m in zip(range(42, 52), ms):
            if m:
                v = m["versions"]; groups.setdefault(f"torch {v['torch']} / numpy {v['numpy']} / sb3 {v['sb3']}", []).append(f"{ag}:{sd}")
    req = read(root, "requirements.txt") or ""
    pins = ", ".join(l.strip() for l in req.splitlines() if "==" in l and l.split("==")[0] in ("torch", "stable-baselines3", "gymnasium", "numpy"))
    mixed = len(groups) > 1
    add("C06", "high" if mixed else "medium", "CONFIRMED" if mixed else "INFO", "Training-environment version drift among shipped checkpoints",
        f"environment groups: { {k: (v if len(v) < 4 else f'{v[0]}..{v[-1]} ({len(v)} models)') for k, v in groups.items()} }; requirements.txt claims 'exact versions used to produce the shipped results' and pins: {pins}. "
        "Local environment NOT inspected here (scripts/x_01_check_env_and_checkpoints.py does that). Retraining under another torch/SB3 build is not bit-reproducible.")
    # ---- C07 convergence & bookkeeping, recomputed from val_curve.csv
    conv = read_csv(root, "results/clean/convergence_summary.csv")
    mh = float(conv[0]["val_baseline_minhop"]) if conv else None
    for ag in ("replace", "enhance"):
        rows = []
        for s in range(42, 52):
            c = read_csv(root, f"results/models/{ag}/seed{s}/val_curve.csv")
            if c:
                rows.append((s, [(int(r["timesteps"]), float(r["val_delivery_ratio"])) for r in c]))
        if not rows:
            continue
        T = max(t for _, c in rows for t, _ in c)
        late = sum(1 for _, c in rows if max(c, key=lambda z: z[1])[0] >= 0.6 * T)
        init_best = [s for s, c in rows if max(c, key=lambda z: z[1])[0] == 0]
        eq_mh = {s: [t for t, v in c if mh is not None and abs(v - mh) < 1e-12] for s, c in rows}
        eq_mh = {s: v for s, v in eq_mh.items() if v}
        init_mean = sum(c[0][1] for _, c in rows) / len(rows); best_mean = sum(max(v for _, v in c) for _, c in rows) / len(rows)
        sev = "high" if ag == "replace" else "high"
        extra = (f"; {len(init_best)} seed(s) {init_best} had their UNTRAINED step-0 policy as the curve maximum, but best.zip excludes step 0 (ValCB.best starts at -1 and step 0 is never saved) -> analyze_clean's 'best' bookkeeping includes it, the saved checkpoint does not" if init_best else "")
        add("C07" + ag[0], sev, "CONFIRMED", f"{ag}: validation-curve evidence ({len(rows)} seeds, budget {T} steps)",
            f"max curve value in last 40% of training for {late}/{len(rows)} seeds; mean init val {init_mean:.4f} -> mean best {best_mean:.4f} (val min-hop {mh}); seeds whose val equals the min-hop baseline EXACTLY at some checkpoint (policy collapsed onto candidate 0): {eq_mh or 'none'}{extra}")
    # ---- C08 calibration realism
    ss = json.loads(read(root, "results/sample_load_summary.json") or "null"); lc = json.loads(read(root, "results/load_calibration_validation.json") or "null")
    if ss and lc:
        mu_s = ss["quantiles"]["max_util"]["0.5"]; mu_l = lc["1.0"]["minhop"]["max_util"]
        add("C08", "medium", "CONFIRMED" if abs(mu_l - mu_s) > 0.1 else "OK", "'load 1.0 calibrated to the shipped samples' holds for total offered volume only",
            f"shipped samples: median max link utilization {mu_s}, 0 drops in 500 samples; simulator load 1.0 (min-hop, 100 validation scenarios): mean max utilization {mu_l}. 60 random pairs concentrate the same volume on fewer paths. Loads > 1 are extrapolations (no calibration evidence).")
    # ---- C09 seed ranges
    try:
        from p2x.config import validate
        validate(x); add("C09", "high", "OK", "Seed ranges disjoint: legacy train/val/test vs new val_v2 / confirmatory / poison", f"validated ranges: {x['seeds']['ranges']}")
    except Exception as e:
        add("C09", "high", "CONFIRMED", "Seed-range validation failed", str(e))
    ec = json.loads(read(root, "results/clean/eval_config.json") or "null")
    if ec:
        add("C10", "high", "INFO", "Existing test set is a development set", f"test seeds {ec['test_seed_base']}..+{ec['n_test']-1} x loads {ec['loads']} were inspected and used for planning; never reuse for tuning; confirmatory uses a fresh range.")
    # ---- C11 terminology
    hits = {}
    for rel in ("configs/phase2_config.json", "src/p2/sim.py", "src/p2/envs.py", "README.md", "STATUS_AND_RESUME.md", "data/PROVENANCE.md"):
        s = read(root, rel)
        if s:
            n = len(re.findall(r"delivery[_ ]ratio|\bdelivered\b|packet[- ]delivery", s, re.I))
            if n:
                hits[rel] = n
    add("C11", "medium", "CONFIRMED" if hits else "OK", "Metric named 'delivery_ratio' (risk of being read as packet delivery ratio)",
        f"occurrences: {hits}. Existing CSV/column names are preserved; all NEW outputs use 'accepted_offered_ratio' (accepted/offered traffic VOLUME).")
    # ---- C12 telemetry-channel coverage
    ev = read(root, "scripts/eval_clean.py") or ""; en = read(root, "src/p2/envs.py") or ""
    bypass = "fn(net, ep, ep.true_util())" in ev
    add("C12", "high", "CONFIRMED" if bypass else "OK", "Baselines bypass the Telemetry channel in the existing evaluation; Telemetry has no per-episode RNG/seed hook",
        f"eval_clean.run_baseline passes ep.true_util() directly={bypass}; Telemetry.observe(true_util) only. Poisoned controls therefore need the new runner (src/p2x/rollout.py) with per-episode seeded telemetry (src/p2x/telemetry.py).")
    # ---- C13 analysis methodology
    an = read(root, "scripts/analyze_clean.py") or ""
    notes = []
    if "ttest_1samp" in an: notes.append("p-values are t-tests over 10 model means only (scenario noise averaged out); scenario CI is a separate bootstrap; no combined interval")
    if "idxmax()" in an: notes.append("best checkpoint step taken by idxmax over a curve that includes untrained step 0")
    if "ci95_lo_models" in an: notes.append("results are reported as fractions, not percentage points")
    add("C13", "medium", "CONFIRMED" if notes else "OK", "Statistical reporting limits in analyze_clean.py", "; ".join(notes) or "none detected")
    # ---- C14 checksums
    sums = read(root, "data/raw/SHA256SUMS.txt")
    if sums:
        res = {}
        for line in sums.splitlines():
            if not line.strip():
                continue
            h, rel = line.split(None, 1); p = root / "data/raw" / rel.strip().lstrip("*")
            res[rel.strip()] = "MISSING" if not p.is_file() else ("OK" if hashlib.sha256(p.read_bytes()).hexdigest() == h else "MISMATCH")
        bad = [k for k, v in res.items() if v == "MISMATCH"]
        miss = [k for k, v in res.items() if v == "MISSING"]
        add("C14", "medium", "CONFIRMED" if bad else ("UNVERIFIED" if miss else "OK"), "data/raw checksums", str(res))
    add("C15", "high", "UNVERIFIED", "Dataset license and identity of the NED", "Provenance file says: fork copy, not verified against the official BNN-UPC archive, dataset license NOT verified, GEANT2 unavailable. Nothing on disk can resolve this; see docs/04_PROVENANCE_ACTIONS.md.")
    add("C16", "high", "UNVERIFIED", "Agent B checkpoint behaviour (action saturation, policy std, weight distribution, route selection)", "Checkpoints are binary and cannot be inspected statically. Run scripts/x_02_diag_checkpoints.py --agent enhance.")
    # ---- report
    w = max(len(c["id"]) for c in CHECKS)
    for c in CHECKS:
        print(f"[{c['status']:<10}] {c['id']:<{w}} ({c['severity']}) {c['title']}\n{'':14}{c['evidence']}")
    out = root / "results/v2/static_audit"; out.mkdir(parents=True, exist_ok=True)
    (out / "static_audit.json").write_text(json.dumps(CHECKS, indent=2))
    print(f"\nwrote {out/'static_audit.json'}")
    if a.strict and any(c["status"] == "CONFIRMED" for c in CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT), help="project root to audit (default: this project)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any CONFIRMED problem")
    main(ap.parse_args())
