# Phase 1 status

Status values: completed · partially completed · blocked · not started. "Validation" lists what was actually executed.

| Task | Status | Evidence / output file | Validation performed | Remaining issue |
|---|---|---|---|---|
| Inventory & preserve originals | completed | `originals/` + `originals/SHA256SUMS.txt` | SHA-256 of every preserved file compared with the uploaded copy: match | `ppo_routing.zip` was **not supplied** (searched the whole filesystem) |
| A. Audit implementation | partially completed | `reports/AUDIT_REPORT.md` | Read all 4 scripts; `diff` main vs backup; instantiated the original PPO to read architecture/hyperparameters; re-ran original scripts (`audit/original_script_rerun/`) | Original model file unavailable → its contents/compatibility cannot be checked |
| B. Validate poisoning | completed | `tests/test_poisoning.py`, `results/logs/pytest_full.log`, Fig 3, `results/episodes.csv` (`pert_*`, `state_flips`) | 37 tests pass (zero-δ = clean bitwise; targeted entries only; actual = min(δ, 1−clean); bounds/NaN; seed reproducibility; true state untouched; counterfactual = poisoned env); equivalence with original env on 10 seeds × 4 configs; action change measured on 10 real trained models | – |
| C. Valid experimental comparison | completed | `results/episodes.csv` (6,000 rows), `results/config.json`, `results/summary_by_condition.csv` | Shortest path, clean PPO, poisoned PPO at δ ∈ {0,.1,.25,.5,1}; identical scenarios across all methods (paired); shortest-path "poisoned" verified identical by measurement | δ=1.0 is an added strength beyond the original sweep (flagged as extension) |
| D. Metrics | partially completed | Report §8 (definitions), `results/summary_*.csv` | All metrics computed on true loads; definitions asserted in `test_metric_definitions_are_consistent_with_true_state` | Packet loss / delivery **not modelled** in the environment; invalid actions are 0 by construction; `mean_utilization` is a path-length proxy |
| E. Reliability | completed | `results/paired_effects_poisoned_minus_clean_ppo.csv`, Fig 1, Fig 4, report §7/§10 | 10 training seeds (42–51) × 100 scenarios (100–199); mean, SD, 95% t-interval across models; sign counts; deterministic pipeline (bitwise-repeatable training test) | n = 10 models; no multiplicity correction; **no significance claim made**; no failed/missing runs |
| F. Reproduce original results | partially completed | `reports/AUDIT_REPORT.md` §4, `tests/test_env_equivalence.py::test_original_csv_internal_consistency`, `audit/original_script_rerun/` | Reported values re-computed from the supplied CSVs: confirmed. Shortest-path row re-executes exactly. Retraining PPO with the original script (seed 42) gives 1.4946/0.6260, not 1.5082/0.6550 | PPO original numbers **not reproducible**; cause undetermined (model missing, stack unknown) |
| F′. Why is the effect small? | completed (inference) | `results/policy_diagnostics.csv`, report §10.6 | Telemetry ablation, exposure and saturation measured on 10 models | Correlational evidence, not causal proof |
| G. Code & documentation | partially completed | `src/`, `scripts/`, `tests/`, `README.md`, `requirements.txt`, `.gitignore` | Whole pipeline executed end-to-end in this environment; package re-extracted to a fresh folder and test-suite re-run there (see final handoff) | A from-scratch virtualenv `pip install -r requirements.txt` was **not** tested (reused the pre-provisioned interpreter) |
| H. Visualizations | completed | `figures/fig1…fig4_*.png` | All four figures rendered and visually inspected (Fig 1 x-label clipping found, fixed and regenerated); axes labelled, captions in report | – |
| Deliverable 1 `PHASE1_REPORT.md` | completed | `PHASE1_REPORT.md` | Numbers cross-checked against `results/*.csv` | Reviewer should skim §10.3 caveats |
| Deliverable 2 source code | completed | `src/`, `scripts/` | Executed | – |
| Deliverable 3 verified outputs | completed | `results/`, `figures/`, `originals/`, `audit/` | Logs in `results/logs/` | – |
| Deliverable 4 tests | completed | `tests/`; `results/logs/pytest_full.log` (37 passed, `RUN_SLOW=1`), `results/logs/pytest_default.log` (36 passed, 1 skipped) | Executed, results as logged | – |
| Deliverable 5 README/requirements/.gitignore | partially completed | root files | Commands executed as written except virtualenv creation | See Task G |
| Deliverable 7 archive | completed | `routing-poisoning-phase1.zip` | Contents listed and extracted tests re-run | **Not published to GitHub** (no publication attempted; local repo-ready package only) |
| ENERO comparison | blocked | – | – | No ENERO code/topologies supplied; no ENERO claim made (only its public abstract was read) |
| Phase 2 | not started | Plan in report §13 | – | Deliberately deferred |
