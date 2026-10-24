# PPO Network Routing Under Observation Poisoning — Phase 1

Reproducible research prototype that measures how falsified link-utilization telemetry affects a PPO routing agent on a small **synthetic** network. It is an independent prototype; it is **not** a reproduction of ENERO.

* Full write-up: [`PHASE1_REPORT.md`](PHASE1_REPORT.md) · Audit of the original code: [`reports/AUDIT_REPORT.md`](reports/AUDIT_REPORT.md) · Task status: [`PHASE1_STATUS.md`](PHASE1_STATUS.md)

## Research question
How does poisoning network telemetry/observations affect the routing decisions and performance of a PPO-based routing agent, and how can this effect be measured reproducibly?

## Model and algorithm
Stable-Baselines3 PPO, `MlpPolicy` (30→64→64 tanh, separate policy/value nets), 20k timesteps, trained on clean observations only. Hyperparameters are those of the original script (`n_steps=256, batch_size=64, lr=3e-4`). Baselines: static shortest-hop routing, and a diagnostic greedy policy that reads the same telemetry.

## Environment
Synthetic 8-node / 13-link topology, capacity 100 per link, 20 demands per episode (volume ~U(10,35)), one demand routed per step by choosing one of its 5 shortest paths. Reward = −mean squared utilization (true loads). See `src/routing_env.py` and report §5. No real-world data.

## Poisoning mechanism
Test-time, additive: `obs[link] = clip(obs[link] + δ, 0, 1)` for links 0 = (0,1) and 3 = (1,3). True loads and reward are untouched. Strengths δ ∈ {0, 0.10, 0.25, 0.50, 1.00}.

## Experimental protocol
10 independently trained PPO models (seeds 42–51) × 100 paired scenarios (seeds 100–199; seeds 100–119 are the original 20) × 5 strengths. Effects are paired differences per model, summarized across models (t-interval, n=10). No significance tests are claimed.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # versions used to produce the results
```
Tested here with Python 3.13.17 on Linux/CPU in a shared interpreter (a from-scratch virtualenv install was **not** tested).

## Run
```bash
python -m pytest                       # 36 tests (+1 slow: RUN_SLOW=1 python -m pytest)
python scripts/train.py                # ~30 s per model -> results/models/ppo_seed{42..51}.zip
python scripts/run_experiment.py       # ~2 min -> results/episodes.csv, results/config.json
python scripts/analyze.py              # tables (results/*.csv, summary_tables.md) + figures/
python scripts/diagnose_policy.py      # results/policy_diagnostics.csv
```
Everything is deterministic given the seeds on the same software stack (training pinned to CPU, 1 thread).
The shipped `results/models/*.zip` let you skip training and run the last three commands directly.

## Layout
```
originals/   unmodified supplied files + SHA256SUMS.txt   (ppo_routing.zip was NOT supplied)
audit/       re-execution of the ORIGINAL scripts (retrained model, rerun CSVs, console summary)
src/         environment and reference policies
scripts/     train / run_experiment / analyze / diagnose_policy
tests/       pytest suite
results/     episodes.csv (6,000 episodes), config.json, summaries, models, logs
figures/     fig1–fig4 (PNG)
reports/     AUDIT_REPORT.md
```
Original results = `originals/*.csv`. New results = everything under `results/` and `figures/`. The two are never mixed in the same file.

## Current results (new; details and uncertainty in the report)
| | max util | mean util |
|---|---|---|
| Shortest path | 1.5335 | 0.6096 |
| PPO clean (mean of 10 models) | 1.4825 | 0.6483 |
| PPO δ=0.25 | 1.4838 | 0.6542 |
| PPO δ=1.00 | 1.5078 | 0.6773 |

Poisoning raises the share of changed decisions with δ (mean 1.4% at δ=0.1 → 13.8% at δ=1) but the mean effect on utilization is small and varies strongly between trained models. The original numbers (PPO 1.5082/0.6550; shortest path 1.5040/0.6237; poisoned 1.5082/0.6557) are confirmed to be in the supplied CSVs, but the PPO ones could not be regenerated (original model missing).

## Limitations
Toy topology; saturated regime (98% of episodes overload a link; no loss model); one non-adaptive attack on two links; 10 models; weak 20k-step agents; no defence; no ENERO comparison. See report §11.

## Future work
Feasible/realistic benchmark first, stronger adaptive attacks, more seeds and unseen topologies, then defences. See report §13.
