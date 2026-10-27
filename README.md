# PPO Network Routing Under Observation Poisoning

A reproducible research project investigating how falsified network telemetry affects reinforcement-learning-based routing decisions and performance.

This repository contains two distinct research stages:
- **Phase 1:** A synthetic proof of concept measuring the effects of test-time observation poisoning on a PPO routing agent.
- **Phase 2:** An NSFNET-bw-based routing benchmark evaluating PPO routing approaches against conventional baselines.

This is an independent research project, not a reproduction of ENERO.

## Research question

How does inaccurate or manipulated link-utilization telemetry affect PPO-based routing decisions and network performance, and how can these effects be measured reproducibly?

## Phase 1 — Synthetic observation poisoning

Phase 1 uses a synthetic 8-node, 13-link network with capacity 100 per link and 20 demands per episode. A Stable-Baselines3 PPO agent (`MlpPolicy`, 30→64→64 tanh architecture) is trained for 20,000 timesteps using clean observations.

At test time, additive perturbations are applied to the observations for links 0 = (0,1) and 3 = (1,3), with strengths δ ∈ {0, 0.10, 0.25, 0.50, 1.00}. True loads and rewards are not directly modified by the poisoning operation.

The protocol includes 10 independently trained models (seeds 42–51), 100 paired scenarios (seeds 100–199), and conventional routing baselines.

The results show that stronger perturbations increase the proportion of changed routing decisions, while utilization effects remain relatively small on average and vary substantially between models. The environment is simplified and does not model realistic packet loss.

### Phase 1 results

| Method | Maximum utilization | Mean utilization |
|---|---:|---:|
| Shortest path | 1.5335 | 0.6096 |
| PPO clean, mean of 10 models | 1.4825 | 0.6483 |
| PPO, δ = 0.25 | 1.4838 | 0.6542 |
| PPO, δ = 1.00 | 1.5078 | 0.6773 |

These figures are the reported Phase 1 results; consult the full report for uncertainty, methodology, and interpretation. The original PPO results could not be regenerated because the original model was missing.

### Phase 1 documentation

- [Full report](PHASE1_REPORT.md)
- [Task status](PHASE1_STATUS.md)
- [Original-code audit](reports/AUDIT_REPORT.md)

### Phase 1 commands

```bash
python -m pytest
python scripts/train.py
python scripts/run_experiment.py
python scripts/analyze.py
python scripts/diagnose_policy.py
```

Install dependencies from `requirements.txt` in the Phase 1 environment before running these commands. Existing trained models can be used to skip training when available.

## Phase 2 — NSFNET-bw routing benchmark

Phase 2 extends the investigation to a 14-node, 21-duplex-link NSFNET-bw-based topology with synthetic traffic scenarios at multiple offered loads.

The benchmark includes two PPO approaches:

- **Agent A:** PPO-based path selection.
- **Agent B:** PPO-based link-weight selection followed by Dijkstra routing.

Conventional baselines are included for comparison, including minimum-hop and shortest-feasible routing where supported by the evaluation artifacts.

### Current findings and limitations

The available diagnostics do not establish that both PPO approaches learn effective routing policies. Agent A's earlier training run had not converged, while Agent B showed little evidence of learning in the reported diagnostics. The evidence does not justify claiming that PPO consistently outperforms the strongest conventional baseline.

Dataset provenance and licensing still require verification.

### Phase 2 status

The `phase2/` directory contains the benchmark code, scripts, tests, and available results. Three diagnostic figures can be regenerated as PNG and PDF files:

- `phase2/figures/fig1_agent_a_training_curves`
- `phase2/figures/fig2_baseline_diagnostics`
- `phase2/figures/fig3_checkpoint_diagnostics`

These figures visualize existing artifacts. They are not independent confirmatory results.

**Telemetry-poisoning experiments have not been completed.** The presence of an evaluation script does not demonstrate that an experiment has been executed or validated. Confirmatory evaluation status must be verified from actual output artifacts and records.

### Phase 2 commands

From the repository root, with Phase 2 dependencies installed:

```bash
python -m pytest phase2/tests -q
python phase2/scripts/x_09_plot_results.py
```

The plotting script uses existing results; it does not train models or run new experiments. Inspect script options and default behavior before starting computationally expensive tasks.

## Repository layout

```text
README.md                    Combined project overview
PHASE1_REPORT.md             Phase 1 research report
PHASE1_STATUS.md             Phase 1 task status
reports/                     Phase 1 audit report
src/                         Phase 1 source code
scripts/                     Phase 1 experiment scripts
tests/                       Phase 1 tests
results/                     Phase 1 results and models
figures/                     Phase 1 figures
phase2/                      Phase 2 benchmark and tooling
phase2/scripts/              Phase 2 experiment and analysis scripts
phase2/tests/                Phase 2 tests
phase2/figures/              Phase 2 diagnostic figures and documentation
```

## Reproducibility and interpretation

Results depend on software versions, random seeds, topology, traffic generation, training procedures, and evaluation protocols. Keep the two phases' code, dependencies, metrics, and result files separate unless a consistent comparison methodology is established.

The Phase 2 accepted/offered ratio measures accepted traffic volume relative to offered traffic volume; it is not a packet delivery ratio.

Do not claim that a defence is effective without completed and validated attack-and-defence experiments. Distinguish exploratory diagnostics from independent confirmatory results.

## Future work

1. Verify Phase 2 dataset provenance and licensing.
2. Verify the confirmatory clean-stage evaluation against its actual artifacts.
3. Establish adequate training and conventional-baseline comparisons.
4. Run and validate the planned telemetry-poisoning experiments.
5. Evaluate potential defences under a documented experimental protocol.
6. Report uncertainty, limitations, and deviations from the planned methodology.