# 01 - Methodology audit of Phase 2 (Stage A, static)

Evidence base: the supplied text dump (code, configs, logs, CSV/JSON results, docs). Not available to me: model checkpoints (binary), the NED, `episodes_clean.csv`, figures,
`reports/*`. Anything needing them is marked **UNVERIFIED**. Re-run `python scripts/x_00_static_audit.py` on your real directory to confirm each row below
(a run on the dump is saved in `docs/evidence/static_audit_on_supplied_dump.json`).

## Verdict
The clean benchmark is usable as a **first-look result**, not as a confirmatory one. Do **not** start poisoning experiments yet. Must be settled first:
(1) Agent B behaves like min-hop and its learning is not demonstrated - decide what it is before attacking it; (2) Agent A is not converged and its gap to a
strong heuristic is not decomposed into "action space" vs "learning"; (3) a fresh, independently seeded test set is needed; (4) the poisoning channel
needs per-episode seeding, baseline controls and invariant checks that the existing evaluation lacks (now provided in `src/p2x`).

## Confirmed issues (verified from the supplied files)
| # | Sev | Finding | Evidence | Impact | Resolution |
|---|---|---|---|---|---|
| 1 | High | **Agent B shows no demonstrated learning.** Mean val at init 0.8551 vs min-hop 0.8546 (val set v1); mean best 0.8591; mean final 0.8470 (below init); best checkpoint in last 40% for only 2/10 seeds; for seeds 48, 49, 51 the *untrained* step-0 policy had the highest curve value. Test: B - min-hop = +0.03, +0.31, +0.19, -0.31 pp (loads 1.0/1.5/2.0/3.0; Holm-adjusted not significant); B - shortest-feasible = -0.76 .. -8.63 pp. 85% of B's routes equal candidate 0 at every load. | `results/clean/convergence_summary.csv`, `models/enhance/*/val_curve.csv`, `paired_ppo_minus_baseline.csv`, `frac_decisions_equal_minhop_path.csv` | Attacks on B would test a near-static min-hop router | `x_02 --agent enhance` (mechanism), `x_03` (null + heuristic), report B as negative/inconclusive |
| 2 | High | **Agent A not converged.** Curve maximum in the last 40% of training for 10/10 seeds; mean best step 145,500/150,000; late gain (last 20% vs 50-70% window) +1.13 pp, 95% CI [0.92, 1.34]. Seeds 42 and 43 collapse *exactly* onto the min-hop validation value (0.854649...) at step 15k. | `convergence_summary.csv`, `replace/seed4{2,3}/val_curve.csv` | Reported A numbers are a 150k-step snapshot | `x_04` pilot with predeclared stopping rule |
| 3 | High | **Mixed training environments.** All Agent A and B seed 42: torch 2.14.1+cu130 / numpy 2.4.4. B seeds 43-51: torch 2.11.0+cu128 / numpy 2.5.2. `requirements.txt` says these are the *exact versions used to produce the shipped results* - false for 9 of 20 checkpoints. | `meta.json` `versions` (all 20) | B is a mixed-build ensemble; bit-reproducibility impossible | Disclose; `x_01` checks load fidelity; `x_08 --exclude-model-seeds 42` gives a homogeneous B sensitivity run |
| 4 | High | **Existing test set is a development set** (seeds 20,000,000..+199, inspected and used for planning). | `eval_config.json`, task statement | Optimism if reused | Fresh range `test_confirmatory_v2` (30,000,000+) with freeze + LOCK |
| 5 | High | **No telemetry channel for baselines / no per-episode RNG.** `eval_clean.run_baseline` passes `ep.true_util()` directly; `Telemetry.observe(true)` has no seed/step hook. | `scripts/eval_clean.py`, `src/p2/envs.py` | Poisoning controls impossible without it | `src/p2x/rollout.py`, `telemetry.py` (existing code untouched) |
| 6 | Med | **Status documents contradict artifacts.** STATUS_AND_RESUME says B training UNFINISHED, eval NOT run, analysis NOT run, report NOT written; disk has 10/10 B `meta.json`, `eval_clean.log` ("wrote 18400 rows"), clean CSVs, convergence/comparison outputs. NOT_INCLUDED says no figures/tables exist. README cites `reports/PHASE2_CLEAN_REPORT.md`, absent from the dump (UNVERIFIED on your disk). | `STATUS_AND_RESUME.md`, `NOT_INCLUDED.md`, `README.md` vs results | Misleading handover | `docs/03_CURRENT_STATUS.md` supersedes; add banner to old files (runbook step) |
| 7 | Med | **"Load 1.0 calibrated to the 500 shipped samples" is true for total offered volume only.** Samples: median max link utilization 0.614, zero drops. Simulator load 1.0 (min-hop, 100 validation scenarios): mean max utilization 0.885, blocking 0.65%. Loads > 1 have no calibration evidence. | `sample_load_summary.json`, `load_calibration_validation.json` | Scenario realism claims must be limited | State as synthetic scenarios; no real-traffic claims |
| 8 | Med | **Metric name `delivery_ratio`** (config, sim docstrings, STATUS) invites reading it as packet delivery ratio; it is accepted/offered traffic volume under flow-level admission. | `phase2_config.json`, `sim.py`, `STATUS_AND_RESUME.md` | Terminology error in papers | All new outputs use `accepted_offered_ratio`; legacy CSVs unchanged |
| 9 | Med | **Analysis bookkeeping/statistics.** "Best step" uses `idxmax` over a curve that includes the untrained step 0, whereas the saved `best.zip` never includes step 0 (ValCB.best starts at -1, step 0 never saved). p-values are t-tests on 10 model means only (scenario noise averaged out); scenario CI is a separate bootstrap; effects reported as fractions. | `analyze_clean.py`, `train_ppo.py` | Cosmetic for conclusions; matters for rigor | `p2x.stats` crossed bootstrap + Bonferroni CIs + Holm, pp units |
| 10 | Med | **Test suite scope.** 21 functions / 23 cases, 23 passed in sandbox logs (consistent). An earlier run had 2 failures (observation bound, empty-network test), fixed and logged. No test covers eval/analysis scripts, checkpoint loading, Agent B under telemetry, stats code. | `tests/test_phase2.py`, `pytest_phase2_run{1,2}.log` | Coverage gap | 70 new tests (`tests/test_x_*.py`) |
| 11 | Med | **Checkpoint selection noise.** Best-of-10 on 100 validation scenarios; B's across-checkpoint spread (roughly 0.1-1 pp per seed) is comparable to the selection noise. | `val_curve.csv` | "best" for B is largely noise | New val set 200 scenarios; confirmatory set independent of selection |
| 12 | High | **Provenance/license unresolved.** NED from a third-party fork, not compared with an official archive; dataset license not verified; GEANT2 unavailable. | `data/PROVENANCE.md` | Cannot publish raw data; topology identity unverified | `docs/04_PROVENANCE_ACTIONS.md` (needs you) |

## Hypotheses (NOT verified; each has a cheap test)
| Hypothesis | Test | What would change |
|---|---|---|
| B's policy std / action saturation / weight spread explain its inertness | `x_02 --agent enhance` | whether B is "near-static" |
| The K=4 candidate set excludes the shortest-feasible path often enough to cap Agent A | `x_03` Q1/Q2 (restricted vs unrestricted heuristic) | whether more training can close the gap or the action space must change |
| A transparent utilization-weight rule works through the same Dijkstra pipeline | `x_03` Q3 (held-out half) | whether B's failure is learning or pipeline design |
| A reaches a plateau by 600k steps | `x_04` pilot | whether to spend ~3 more hours of CPU |
Already informative: A is below `greedy_widest` (same 4 candidates) by 0.13/2.21/2.17 pp at loads 1.0/1.5/2.0 on the first-look set, so the action set already contains better policies than A reaches.

## Resolved / verified OK
23 legacy tests pass in the original logs; scenario seed ranges are disjoint (re-verified against code by `tests/test_x_config_seeds.py`); reward is computed from true state (legacy test); admission never exceeds capacity (legacy tests); raw-data checksum list is internally consistent (verification of the NED itself needs your disk).
