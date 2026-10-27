# Verification status of this deliverable (what was actually executed)

Environment of my checks: Python 3.12.3 sandbox with numpy, networkx, gymnasium, pandas, scipy, pytest. **torch and stable-baselines3 were NOT available** (an install attempt was started earlier and then stopped at your request; neither imports). The **real NED was not available**; checks that need a network used a *stand-in* topology (standard 14-node/21-link NSFNET adjacency, written only into the sandbox, **not shipped**). Stand-in runs verify code logic only; none of their numbers is evidence and none appears in these documents.
| Item | State |
|---|---|
| Legacy 22 tests (all but the NED checksum test) on stand-in | executed, passed (logic only) |
| 70 new tests on stand-in (1 skipped: "accepts the real NED") | executed, passed (logic only); includes a mutation check showing the invariant tests fail for an aliasing telemetry |
| x_00 static audit on the supplied dump | executed on real documents/logs/results; output saved in `docs/evidence/` |
| x_03, x_06 (smoke/freeze/lock/resume), x_07 (dry/smoke), x_08, x_02 (stub models) | executed through `execute()` with stub models on stand-in; passed (logic only) |
| `main()` refusing a non-recorded topology | executed: exit code 2 |
| All 9 scripts import and `--help` exits 0 | executed, passed |
| x_01, x_02 with real SB3 checkpoints; x_04 training; x_06/x_07 with real models; real-NED runs of anything | **not executed** (needs your environment) |
| `src/p2x/train_long.py::train_agent_a` (PPO training, resume) | **not executed**; reviewed against SB3 API only. Pure helpers (stop rule, best-selection) are tested |
| Any scientific result on the real topology | **none produced** |
