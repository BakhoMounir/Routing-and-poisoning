# p2x - audit/validation overlay for the Phase 2 RL-SDN routing project

Unzip into the project root (nothing existing is overwritten). Read `docs/01_METHODOLOGY_AUDIT.md` first, then `docs/02_EXPERIMENT_PROTOCOL.md`.
Everything new is configured in `configs/x_experiments.json` and writes to `results/v2/` only. **Nothing here has been run on your real topology or checkpoints** (see `docs/VERIFICATION_STATUS.md`).
Metric everywhere: `accepted_offered_ratio` = accepted/offered traffic VOLUME (not packet delivery). Poisoning = observation-only perturbation (docs/THREAT_MODEL.md).

## Runbook (run from the project root, project venv active; nothing installs packages)
```bash
# 0. Static checks (seconds, stdlib only)
python scripts/x_00_static_audit.py                      # -> results/v2/static_audit/static_audit.json
python -m pytest tests -q                                # legacy 23 + new 71 tests, on YOUR real NED. Expect all pass.
                                                         # (the NED-hash tests must NOT skip/fail; a skip means the NED differs from the recorded hash)
# 1. Cheap read-only diagnostics
python scripts/x_01_check_env_and_checkpoints.py         # ~2-5 min -> results/v2/env_check/checkpoint_integrity.csv (REPRODUCED / FLOAT-LEVEL / DIVERGES)
python scripts/x_03_diag_baselines.py                    # <1 min   -> results/v2/diag_baselines/{decisions,heuristic_selection,...}.json
python scripts/x_02_diag_checkpoints.py --agent enhance  # ~5-15 min -> results/v2/diag_checkpoints_enhance/{per_model.csv,summary.json}
python scripts/x_02_diag_checkpoints.py --agent replace
# 2. Decide with docs/02 (E2/E3 rules). Then, ONLY if justified:
python scripts/x_04_train_agent_a.py                     # DRY RUN: prints plan + cost, trains nothing
python scripts/x_04_train_agent_a.py --preset pilot --run                       # 3 seeds, 600k max, stopping rule
python scripts/x_04_train_agent_a.py --seeds 45 46 47 48 49 50 51 --run         # only if the pilot rule says go (parallelizable per seed)
# 3. Freeze, then confirmatory (fresh seeds; LOCK created; no tuning after this)
python scripts/x_06_confirmatory_eval.py --smoke         # optional pipeline check on development seeds
python scripts/x_05_freeze.py                            # dry run; add --yes when the checklist in docs/02 is met
python scripts/x_05_freeze.py --yes
python scripts/x_06_confirmatory_eval.py                 # -> results/v2/confirmatory_v1/{episodes.csv,parts/,LOCK.json,manifest.json}
python scripts/x_08_analyze.py --stage confirmatory      # -> .../analysis/{summary_by_method_load,primary_comparisons_pp,secondary_..._EXPLORATORY}.csv
python scripts/x_08_analyze.py --stage confirmatory --exclude-model-seeds 42   # Agent B environment-drift sensitivity
# 4. Poisoning (only after step 3 completed)
python scripts/x_07_poison_eval.py --preset smoke --run  # pipeline/invariant check; numbers uninterpretable
python scripts/x_07_poison_eval.py --preset pilot        # dry run (plan); add --run
python scripts/x_07_poison_eval.py --preset full --run --targets A_600k      # split across processes by --targets; parts are resumable
python scripts/x_08_analyze.py --stage poison --preset full
```
Safe defaults: `x_04` and `x_07` are dry runs without `--run`; `x_06` refuses without a valid freeze; every script refuses a NED whose SHA-256 differs from the recorded one; completed outputs are not overwritten without `--overwrite`; importing `p2x` has no side effects.
Every run writes `manifest.json` (config hash, seeds, code hash/git, package versions, input/output hashes, status).

## Current research status and scope

This directory contains the Phase 2 audit and experiment framework. The original verification status describes the state of this framework at the time of that audit; it should not be interpreted as a statement that no subsequent training work has occurred.

Subsequent work has reported Agent A PPO training runs for seeds 42–51, with seed 43 reported as having the best validation score (0.903064 at step 456,000). These are validation results, not independent confirmatory-test results.

Before treating the Phase 2 framework as fully verified, the training artifacts and metadata must be reconciled with the results reported above. The independent clean evaluation and telemetry-poisoning experiments must be tracked separately and must not be described as completed until their outputs and verification records exist.

The reported `accepted_offered_ratio` measures accepted/offered traffic volume; it is not packet delivery ratio.
