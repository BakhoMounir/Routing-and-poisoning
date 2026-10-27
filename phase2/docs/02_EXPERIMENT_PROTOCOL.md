# 02 - Economical experiment plan and gates

All costs are **indicative extrapolations** from the legacy logs (sandbox, 1 core: A 150k steps ~195-320 s; eval ~0.04 s/episode A, ~0.02 s B; baselines ~2 ms). Measure on your machine; nothing below was run on real data by me.
Config for everything: `configs/x_experiments.json` (frozen by `x_05` before Stage E).

| Id | Stage | Question | Minimum useful run (command) | Cost | Information gain | Decision rule |
|---|---|---|---|---|---|---|
| E0 | A | Which audit findings hold on my disk? | `x_00_static_audit.py` | seconds | High: confirms/refutes table in doc 01 | Fix docs before anything else |
| E1 | B | Do shipped checkpoints load and reproduce under my torch/SB3? | `x_01_check_env_and_checkpoints.py` | ~2-5 min | High: gates every use of old checkpoints | Any `DIVERGES` (>1e-3 val diff): do not evaluate that model locally; investigate build |
| E2 | B | What is Agent B actually doing (std, saturation, weights, route == unit-weight route)? What does A do (collapse, confidence)? | `x_02_diag_checkpoints.py --agent enhance` / `--agent replace` | ~5-15 min each | High for B | If >= 8/10 B models choose the unit-weight route >= 95% of decisions *and* are ratio-equivalent (+-0.2 pp) to unit-weight Dijkstra: B = negative result, no B retraining; B is reported as a near-static router in Stage F |
| E3 | B/C | (Q1) Is K=4 binding? (Q2) coverage of shortest-feasible path by candidates, (Q3) does a transparent util-weight rule work through the Dijkstra pipeline? | `x_03_diag_baselines.py` | < 1 min | High, decides E4 framing and B redesign | Q1 "positive": action space limits A -> document, do not expect training alone to match shortest-feasible. Q3 "positive" (held-out half): B's failure is not pipeline capacity -> a *new* proposal (policy as residual on the heuristic) may be justified - **not implemented, needs your approval**. Q3 not positive: stop investing in B |
| E4 | D | Does Agent A keep improving to 600k under the predeclared rule? | `x_04_train_agent_a.py --preset pilot --run` (3 seeds) | ~45-75 min total (1 core) | High | Proceed to E5 iff pilot mean best val (val_v2) exceeds legacy-A on the same set (`x_02 --agent replace`) by >= 0.5 pp. Else keep legacy A, state "not converged" |
| E5 | D | Same, all 10 seeds | `x_04_train_agent_a.py --seeds 45 46 47 48 49 50 51 --run` (parallelizable) | ~2-3.5 h | Needed for P1/P2 | none; completes the ensemble |
| E6 | E | Confirmatory clean comparison on a fresh set | `x_05_freeze.py --yes` -> `x_06_confirmatory_eval.py` -> `x_08_analyze.py --stage confirmatory` | ~35-60 min | The only confirmatory evidence | Primary family P1-P4 x 4 loads, Holm; no tuning afterwards |
| E7a | F | Does the poisoning pipeline run and keep invariants? | `x_07_poison_eval.py --preset smoke --run` | ~seconds | Pipeline check (numbers uninterpretable) | Stop on any InvariantViolation |
| E7b | F | Pilot effect sizes | `x_07_poison_eval.py --preset pilot --run` | ~12 min (plan) | Medium | Only after E6; matrix is NOT edited afterwards |
| E7c | F | Full matrix | `x_07_poison_eval.py --preset full --run` (split with `--targets`) | ~11 h single core (plan) | Primary Stage F evidence | Claim policy in THREAT_MODEL |

## What is exploratory vs confirmatory
* **Exploratory (development set `val_v2`, may steer decisions, never cited as evidence):** all of E2, E3, E4 validation scores, heuristic selection. In E3 the heuristic is *selected on the selection half* (scenario idx < n/2) and its statistics are computed on the *disjoint estimation half*; the full-set grid numbers are labelled OPTIMISTIC.
* **Confirmatory (fresh seeds 30,000,000..30,000,499, evaluated once after freeze):** primary family per load: P1 A_600k - shortest_feasible; P2 A_600k - minhop; P3 B_legacy - unit_weight_dijkstra (B's correct null); P4 util_weight_dijkstra - unit_weight_dijkstra (independent test of the heuristic selected on val_v2). Secondary rows (e.g. A vs A_legacy, K-restricted baseline) are exploratory.
* The first-look set (seeds 20,000,000..+199) is never used for decisions again (role `legacy_reference`).

## Statistics
Effects in percentage points. For every comparison three intervals: scenario-only (conditional on these trained models), across-model-seed (t), crossed bootstrap (models and scenarios resampled; slightly conservative). Primary: Holm-adjusted bootstrap p, Bonferroni-adjusted crossed CI. Model-vs-model contrasts treat the reference ensemble as its per-scenario average (independent trainings are not paired). Loads share scenario seeds, so each load is analysed separately. With only 10 training seeds the across-seed uncertainty is itself imprecise.

## Freeze checklist (before `x_05_freeze.py --yes`)
- [ ] `pytest tests -q` passes locally on the real NED; `x_00` and `x_01` reviewed; no `DIVERGES`.
- [ ] E3 run and `heuristic_selection.json` reviewed; E4/E5 outcome recorded (or A_600k declared absent).
- [ ] `configs/x_experiments.json` (including the attack matrix) final. Any later edit voids the freeze (hash-checked).
## Acceptance checklist per stage
| To enter | Must hold |
|---|---|
| Stage B | tests pass; `x_00` run; topology hash guard passes on your NED |
| Stage C/D | E1 has no DIVERGES (or affected models excluded and noted); E2/E3 outputs read |
| Stage E | freeze written; LOCK absent; `x_04` runs (if used) all show `meta.json` with `stop_reason` recorded |
| Stage F | confirmatory manifest `status=completed`; `invariants.all_replay_ok` and `all_caps_ok` true; `max_delta_over_budget <= 1e-9` |
