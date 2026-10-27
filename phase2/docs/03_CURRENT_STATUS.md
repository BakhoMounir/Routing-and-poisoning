# 03 - Current status (supersedes STATUS_AND_RESUME.md)

Derived from the supplied dump. Re-verify with `scripts/x_00_static_audit.py`.

| Item | Status | Evidence |
|---|---|---|
| Topology parse + validation (14 nodes, 21 duplex links, 18x10 + 3x40 CU) | done | `topology_validation.json` |
| Agent A, 10 seeds x 150k steps | done (not converged) | 10 meta.json, `train_replace.log` |
| Agent B, 10 seeds x 150k steps | **done** (STATUS_AND_RESUME says unfinished - stale). Seed 42 under torch 2.14.1, seeds 43-51 under torch 2.11.0 | meta.json, `train_enhance.log`, `resume.out` |
| First-look clean evaluation (200 scen x 4 loads x 20 models + 3 baselines = 18,400 rows) | done | `eval_clean.log`, `eval_config.json` |
| Clean analysis tables / figures | done (STATUS says not run - stale) | `results/clean/*.csv`, figures |
| Clean-stage report | not in the dump; README cites `reports/PHASE2_CLEAN_REPORT.md` - UNVERIFIED | - |
| Poisoning | none implemented in Phase 2 except an identity telemetry hook; Stage F now *prepared* (not run) | - |
| GEANT2 | blocked | PROVENANCE.md |
| New extension layer `p2x` + 9 scripts + 70 tests | implemented; see `VERIFICATION_STATUS.md` for what was executed | - |

Suggested one-line banner for the old files (manual, optional):
`echo '> SUPERSEDED: see docs/03_CURRENT_STATUS.md (Agent B finished, evaluation and analysis done).' | cat - STATUS_AND_RESUME.md > /tmp/s && mv /tmp/s STATUS_AND_RESUME.md`
