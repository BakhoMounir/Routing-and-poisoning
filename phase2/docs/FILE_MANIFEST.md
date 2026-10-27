# File manifest

Overlay to unzip into the Phase 2 project root. **No existing file is modified or overwritten.**
| Path | Status | Purpose |
|---|---|---|
| `README_P2X.md` | new | overview + runbook (project README untouched) |
| `configs/x_experiments.json` | new | all new experiment parameters, seed registry, attack matrix, decision thresholds |
| `src/p2x/config.py` | new | config validation, seed-range disjointness, role guard, topology hash guard, protected paths |
| `src/p2x/manifest.py` | new | run manifest (config, seeds, git/code hash, env versions, inputs/outputs, status) |
| `src/p2x/policies.py` | new | K-restricted shortest-feasible, unit-weight Dijkstra, utilization-weight heuristic (non-learned) |
| `src/p2x/telemetry.py` | new | additive, budgeted observation-only perturbations |
| `src/p2x/rollout.py` | new | runners with telemetry channel, replay invariant, counterfactual flips |
| `src/p2x/stats.py`, `analysis.py` | new | crossed bootstrap, Holm, Bonferroni CIs; confirmatory/poison tables |
| `src/p2x/train_long.py` | new | resumable Agent A training + stopping rule (needs torch/SB3) |
| `src/p2x/parts.py`, `cli.py` | new | resumable part files, freeze verification; CLI helpers |
| `scripts/x_00 ... x_08` | new | static audit, integrity, diagnostics, no-training diagnostics, training, freeze, confirmatory, poisoning, analysis |
| `tests/x_helpers.py`, `tests/test_x_*.py` | new | 71 tests (70 pass + 1 conditional skip) |
| `docs/01..04, THREAT_MODEL, VERIFICATION_STATUS, FILE_MANIFEST, evidence/` | new | audit, protocol, status, provenance, threat model |
| `src/p2/*`, `scripts/{train_ppo,eval_clean,analyze_clean,...}.py`, `tests/test_phase2.py`, `tests/conftest.py`, `configs/phase2_config.json`, `results/*`, `data/*`, `figures/*`, `requirements.txt` | **preserved, untouched** | `tests/conftest.py` (existing) puts `src/` on the path for the new tests |
| `STATUS_AND_RESUME.md`, `NOT_INCLUDED.md`, `README.md` (project) | preserved; optional banner (docs/03) | stale |
New outputs go only to `results/v2/` (guarded against legacy paths).
