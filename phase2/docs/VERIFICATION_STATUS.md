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

## Confirmatory clean-stage evaluation — 2026-10-28/29

A later confirmatory evaluation was completed in the external
`routing-poisoning-phase2-handover` directory. The final audit reported
**35 checks passed and 0 failed**.

Recorded verification evidence:

- Evaluation manifest status: `completed`.
- Evaluation episode rows: 72,000.
- Four tested loads: 1.0, 1.5, 2.0, and 3.0, with 500 scenarios per load.
- Episode CSV SHA-256:
  `e5257ccfa3e8b56661f35833fc894ee9b0bda06c2adf79f25552d442d7f83b73`.
- Frozen configuration SHA-256:
  `2c2a530d649f947b791d33c7c0de58ad9da0e138e9532ca159cf1b24e7eb34ec`.
- Freeze record SHA-256:
  `0747e439645597b7eec67c00d02375406e33f2ee03b0caafd98183f272f97733`.
- The episode data passed the recorded schema, uniqueness, ratio-tolerance,
  scenario-coverage, replay, and capacity checks.
- Analysis output hashes matched the analysis provenance record.

**Scope limitation:** this verifies consistency of the recorded evaluation
artifacts and checks implemented by the audit. It does not independently
establish scientific validity, resolve dataset licensing, or demonstrate
end-to-end reproducibility from the Git repository alone.

The evaluation manifest records `git: null`. The checkpoints were produced
under multiple recorded PyTorch builds. Dataset identity/provenance and
redistribution rights remain unresolved. The full confirmatory outputs and
freeze record remain in the external handover directory and are not thereby
made part of this Git commit.

The earlier sandbox/stand-in verification above is retained as a historical
record of the checks performed at that stage; it should not be interpreted
as describing this later confirmatory run.

Poisoning experiments remain on hold pending review and approval of the
clean-stage results and resolution or explicit handling of the provenance
limitations.
