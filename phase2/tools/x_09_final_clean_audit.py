#!/usr/bin/env python3
"""Read-only final audit of the Phase 2 confirmatory clean benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Read-only audit of the Phase 2 confirmatory clean benchmark."
)
parser.add_argument(
    "--root",
    required=True,
    help="Handover root containing results/v2/freeze.json and confirmatory_v1/",
)
parser.add_argument(
    "--report",
    help="Optional report output path; defaults to results/v2/clean_stage_final_audit.txt under --root.",
)
args = parser.parse_args()

ROOT = Path(args.root).resolve()
RUN = ROOT / "results/v2/confirmatory_v1"
ANALYSIS = RUN / "analysis_v2_valid_tests"
REPORT = (
    Path(args.report).resolve()
    if args.report
    else ROOT / "results/v2/clean_stage_final_audit.txt"
)

checks = []
notes = []

def check(label, ok, evidence):
    checks.append((bool(ok), label, str(evidence)))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    return json.loads(path.read_text())

# Required artifacts
required = [
    ROOT / "results/v2/freeze.json",
    RUN / "episodes.csv",
    RUN / "manifest.json",
    RUN / "LOCK.json",
    OVERLAY / "configs/x_experiments.json",
    OVERLAY / "src/p2x/stats.py",
    OVERLAY / "src/p2x/analysis.py",
    OVERLAY / "scripts/x_08_analyze.py",
    ANALYSIS / "analysis_provenance.json",
    ANALYSIS / "primary_comparisons_pp.csv",
    ANALYSIS / "secondary_comparisons_pp_EXPLORATORY.csv",
    ANALYSIS / "summary_by_method_load.csv",
]
for p in required:
    check(f"Required file: {p.name}", p.is_file(), p)

freeze = load_json(ROOT / "results/v2/freeze.json")
manifest = load_json(RUN / "manifest.json")
lock = load_json(RUN / "LOCK.json")
config = OVERLAY / "configs/x_experiments.json"
episodes = RUN / "episodes.csv"
prov = load_json(ANALYSIS / "analysis_provenance.json")

check(
    "Confirmatory config matches freeze",
    sha(config) == freeze.get("config_sha256"),
    f"actual={sha(config)} expected={freeze.get('config_sha256')}",
)
check(
    "Confirmatory lock matches freeze",
    lock.get("freeze_config_sha256") == freeze.get("config_sha256"),
    f"lock={lock.get('freeze_config_sha256')}",
)
check(
    "Evaluation manifest completed",
    manifest.get("status") == "completed",
    manifest.get("status"),
)
n_rows = manifest.get("n_rows", manifest.get("extra", {}).get("n_rows"))
check("Evaluation manifest reports 72,000 rows", n_rows == 72000, n_rows)

git_revision = manifest.get("git")
if git_revision is None:
    notes.append(
        "WARNING: evaluation manifest has git=null; this is a provenance "
        "limitation, not a failed benchmark execution."
    )
else:
    check("Git revision recorded for evaluation", True, git_revision)

csv_hash = sha(episodes)
expected_csv_hash = manifest.get("outputs", {}).get(str(episodes))
check(
    "Episode CSV hash matches evaluation manifest",
    expected_csv_hash == csv_hash,
    f"actual={csv_hash} expected={expected_csv_hash}",
)

# Validate episode rows and the evaluation invariants.
with episodes.open(newline="") as f:
    rows = list(csv.DictReader(f))

check("Episode CSV has 72,000 rows", len(rows) == 72000, len(rows))
required_cols = {
    "method", "model_seed", "load", "scenario_idx",
    "scenario_seed", "accepted_offered_ratio",
    "replay_ok", "caps_ok",
}
columns = set(rows[0]) if rows else set()
check("Episode CSV required columns", required_cols <= columns,
      f"missing={sorted(required_cols - columns)}")

keys = [
    (r["method"], r["model_seed"], r["load"], r["scenario_idx"])
    for r in rows
]
check("No duplicate episode keys", len(keys) == len(set(keys)),
      f"rows={len(keys)} unique_keys={len(set(keys))}")

import math

TOL = 1e-12
ratio_ok = True
ratio_issues = 0
volume_issues = 0
max_ratio_error = 0.0
max_excess = 0.0

for r in rows:
    try:
        ratio = float(r["accepted_offered_ratio"])
        offered = float(r["offered_volume"])
        accepted = float(r["accepted_volume"])

        if not all(math.isfinite(x) for x in (ratio, offered, accepted)):
            ratio_issues += 1
            continue
        if offered <= 0 or accepted < -TOL or accepted > offered + TOL:
            volume_issues += 1

        expected = accepted / offered
        error = abs(ratio - expected)
        max_ratio_error = max(max_ratio_error, error)
        max_excess = max(max_excess, ratio - 1.0, 0.0)

        if ratio < -TOL or ratio > 1.0 + TOL or error > TOL:
            ratio_issues += 1
    except (ValueError, TypeError, ZeroDivisionError):
        ratio_issues += 1

ratio_ok = ratio_issues == 0 and volume_issues == 0
check(
    "Acceptance ratio and volume relationships valid within numerical tolerance",
    ratio_ok,
    f"tolerance={TOL}; ratio issues={ratio_issues}; "
    f"volume issues={volume_issues}; max ratio error={max_ratio_error:.3g}; "
    f"max excess above 1={max_excess:.3g}",
)

load_counts = {}
scenario_sets = {}
for r in rows:
    load = float(r["load"])
    load_counts[load] = load_counts.get(load, 0) + 1
    scenario_sets.setdefault(load, set()).add(int(r["scenario_idx"]))
expected_loads = {1.0, 1.5, 2.0, 3.0}
check("Four expected loads", set(load_counts) == expected_loads, sorted(load_counts))
check("Exactly 500 scenarios per load",
      all(len(scenario_sets.get(x, set())) == 500 for x in expected_loads),
      {k: len(v) for k, v in scenario_sets.items()})
check("Scenario indices 0–499 per load",
      all(scenario_sets.get(x) == set(range(500)) for x in expected_loads),
      "checked all four loads")

replay_ok = all(r.get("replay_ok", "").lower() in {"true", "1"} for r in rows)
caps_ok = all(r.get("caps_ok", "").lower() in {"true", "1"} for r in rows)
check("Replay invariant passes", replay_ok, "all rows")
check("Capacity invariant passes", caps_ok, "all rows")

# Analysis provenance and exact output hashes.
check(
    "Analysis provenance benchmark freeze hash matches",
    prov.get("benchmark_freeze_sha256") == sha(ROOT / "results/v2/freeze.json"),
    f"recorded={prov.get('benchmark_freeze_sha256')}",
)
for name, expected in prov.get("analysis_outputs", {}).items():
    p = ANALYSIS / name
    check(
        f"Analysis output hash: {name}",
        p.is_file() and sha(p) == expected,
        f"actual={sha(p) if p.is_file() else 'MISSING'} expected={expected}",
    )

# Check comparison-table sizes and key statistical columns.
with (ANALYSIS / "primary_comparisons_pp.csv").open(newline="") as f:
    primary = list(csv.DictReader(f))
with (ANALYSIS / "secondary_comparisons_pp_EXPLORATORY.csv").open(newline="") as f:
    secondary = list(csv.DictReader(f))
with (ANALYSIS / "summary_by_method_load.csv").open(newline="") as f:
    summary = list(csv.DictReader(f))

check("16 primary comparison rows", len(primary) == 16, len(primary))
check("32 exploratory comparison rows", len(secondary) == 32, len(secondary))
check(
    "Primary adjusted p-values present",
    bool(primary) and all(r.get("p_holm", "") not in {"", "nan", "NaN"} for r in primary),
    "p_holm checked",
)
check(
    "Primary test methods present",
    bool(primary) and all(r.get("test_method", "") for r in primary),
    sorted({r.get("test_method", "") for r in primary}),
)
check("Summary table non-empty", bool(summary), len(summary))

# Preserve explicit known interpretation/provenance caveats.
notes.extend([
    "Training checkpoints came from multiple recorded PyTorch builds.",
    "Dataset identity/provenance and redistribution rights remain unresolved.",
    "Statistical tests and crossed-bootstrap confidence intervals use distinct inferential procedures.",
    "Agent B has no demonstrated advantage over unit_weight_dijkstra in the current results.",
    "A_600k beats minhop but does not consistently beat the strongest conventional baselines.",
    "Do not begin poisoning experiments until the clean-stage results are reviewed and approved.",
])

lines = [
    "PHASE 2 CLEAN-STAGE FINAL AUDIT (REVISED)",
    f"Run directory: {RUN}",
    f"Analysis directory: {ANALYSIS}",
    "",
]
for ok, label, evidence in checks:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {label} — {evidence}")
lines += ["", "PROVENANCE / INTERPRETATION NOTES"]
lines += [f"[NOTE] {n}" for n in notes]
lines += [
    "",
    f"Checks passed: {sum(ok for ok, _, _ in checks)}/{len(checks)}",
    f"Checks failed: {sum(not ok for ok, _, _ in checks)}",
    f"Episode CSV SHA-256: {csv_hash}",
]

REPORT.write_text("\n".join(lines) + "\n")
print(REPORT.read_text())
if any(not ok for ok, _, _ in checks):
    raise SystemExit(1)
