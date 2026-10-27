# Phase 2 Figures

These figures are generated from existing Phase 2 result artifacts. They are diagnostic visualizations, not independent confirmatory results.

## Regenerate

From the repository root, with the Phase 2 Python dependencies installed:

    python phase2/scripts/x_09_plot_results.py

The script writes PNG (300 DPI) and PDF versions to this directory. It does not train models or run evaluation experiments.

## Figures

- `fig1_agent_a_training_curves`: validation histories for available Agent A seeds.
- `fig2_baseline_diagnostics`: accepted/offered traffic-volume ratio by load for methods present in the baseline diagnostic data.
- `fig3_checkpoint_diagnostics`: best-checkpoint diagnostic scores across available model seeds.

The shaded training region shows plus/minus one across-seed standard deviation. Baseline error bars are descriptive normal-approximation intervals based on episode-level standard errors and do not account for model-seed uncertainty. Checkpoint error bars show across-seed standard deviation, not confidence intervals.

The accepted/offered ratio is a traffic-volume acceptance metric, not a packet delivery ratio. These diagnostic figures are not confirmatory results. They do not demonstrate that telemetry-poisoning experiments have been completed.
