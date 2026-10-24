# Audit report — `torch-routing-security` prototype (Task A / F)

Scope: the five files supplied in the attachments. **`ppo_routing.zip` was listed in the brief but was NOT among the attachments**, so the original trained model could not be inspected.
All items below are tagged **[VERIFIED]** (checked by reading code, running code, or computing from the supplied CSVs), **[ASSUMPTION]**, **[BUG]**, or **[OPEN]**.

## 1. File inventory and provenance

| File | SHA-256 (first 12) | Role (verified) |
|---|---|---|
| `routing_poisoning.py` | `ba29ce77c6cd` | Environment + PPO training + clean/poisoned evaluation (`main()` writes `results.csv`, `ppo_routing.zip`) |
| `routing_poisoning_backup.py` | `8a2c4ff5703a` | Older version of the same file; differs only in `evaluate()` (see §2) |
| `evaluate_poisoning_levels.py` | `537aa186d4c1` | Loads a saved PPO and evaluates δ ∈ {0, .10, .25, .50} on seeds 100–119; writes `poisoning_strength_results.csv` |
| `results.csv` | `1eae09bfd7d8` | 80 rows, 6 columns (no `delta` column; condition ∈ {clean, poisoned}) |
| `poisoning_strength_results.csv` | `90c59dacf55a` | 80 rows, 8 columns incl. per-step `actions` |
| `ppo_routing.zip` | — | **MISSING** |

Full hashes: `originals/SHA256SUMS.txt`. Originals are preserved unmodified in `originals/`.

## 2. Backup vs current script  [VERIFIED by `diff`]

Identical environment, PPO setup, seeds and metrics. Only `evaluate()` differs:
backup: `evaluate(model, poison, ...)`, hard-coded default δ, label `"poisoned"`, no `delta` column;
current: `evaluate(model, poison, delta, ...)`, label `"delta_0.25"`, adds a `delta` column.
**Decision:** retain the backup in `originals/` (it is the version that matches the supplied `results.csv`, see B2). Not needed in the Phase 1 source tree.

## 3. Verified description of the system

* **Data/environment [VERIFIED]:** fully synthetic. 8 nodes, 13 undirected links (confirmed from `EDGES`), uniform capacity 100, unit link weights (hop count), 20 demands per episode, source/destination uniform without replacement, volume ~ U(10, 35). The file itself says "toy topology for testing the experimental pipeline. Replace with a verified research topology before reporting results." No external dataset is used.
* **Episode [VERIFIED]:** one demand routed per step; `terminated` after 20 steps; never truncated. Loads accumulate, never decay, no departures.
* **Action [VERIFIED]:** `Discrete(5)` = index into the 5 shortest simple paths (by hop count) for the current demand. All 56 ordered node pairs have ≥ 5 simple paths, so path padding never occurs and every action is valid.
* **Observation [VERIFIED]:** 30 floats in [0,1] = 13 link utilizations (clipped to [0,1]) + one-hot source (8) + one-hot destination (8) + volume/100.
* **Reward [VERIFIED]:** `−mean_e (load_e/100)²` at every step, computed from TRUE loads. It is a cumulative-state signal, not the marginal effect of the current decision.
* **PPO [VERIFIED by instantiating the original code]:** Stable-Baselines3 PPO, `MlpPolicy`; separate policy and value MLPs each 30→64→64 (tanh), 5-logit action head; `n_steps=256, batch_size=64, lr=3e-4`, SB3 defaults otherwise (`gamma=.99, gae_lambda=.95, clip=.2, n_epochs=10, ent_coef=0, vf_coef=.5`), 20,000 timesteps (≈1000 episodes), seed 42, `device="auto"`.
* **Poisoning [VERIFIED]:** at every step the delivered utilization of link indices 0 = (0,1) and 3 = (1,3) is replaced by `clip(obs + δ, 0, 1)`. Only the observation is altered; simulator loads and reward are untouched. It is state-independent, can only inflate perceived utilization, targets 2 of 13 links, and is applied at **test time only** (the agent is trained on clean observations). This is observation poisoning/evasion, not training-data or parameter poisoning.
* **Shortest-path baseline [VERIFIED]:** `action = 0`, i.e. a minimum-hop path, with no use of loads or telemetry.
* **Train/test [VERIFIED]:** same topology and demand distribution; evaluation scenarios are reset seeds 100–119 (original). Exact overlap with the training RNG stream was not checked.
* **ENERO [VERIFIED absence]:** the attachments contain no ENERO source, topology, or mention. The prototype is an independent toy. ENERO itself (Almasan et al., arXiv:2109.10883) is described by its abstract as a DRL agent with an integrated GNN followed by a local-search refinement, which is architecturally different from this MLP-PPO toy. **No ENERO reproduction is claimed.**

## 4. Reproduction of the originally reported numbers (Task F)

**Values confirmed from the supplied CSVs [VERIFIED]** (20 scenarios, one model):

| | mean max util | mean link util |
|---|---|---|
| PPO clean | 1.5082 | 0.6550 |
| Shortest path | 1.5040 | 0.6237 |
| PPO poisoned (δ=0.25) | 1.5082 | 0.6557 |

The supplied CSVs are internally consistent: `results.csv` poisoned-PPO rows equal the δ=0.25 rows of `poisoning_strength_results.csv`, clean rows equal δ=0 rows (bit-identical).

**Re-execution [VERIFIED]:** running the unmodified current `routing_poisoning.py` here (same seed 42, library versions in `requirements.txt`) gave PPO clean 1.4946 / 0.6260 — **not** the original 1.5082 / 0.6550. The shortest-path row reproduced exactly (1.5040 / 0.6237), which validates the environment, scenario generation and baseline. So the PPO numbers are **not reproducible from the code alone**; the original model file is needed (missing) or the original library/hardware stack differs. The cause cannot be determined from the evidence. Originals are kept untouched; the rerun outputs are in `audit/original_script_rerun/`.

**What the original CSVs show about the attack [VERIFIED]:** δ=0.10 changed 0/400 decisions; δ=0.25 and δ=0.50 each changed exactly 1/400 (seed 105, step 5, action 0→1); δ=0.25 and 0.50 results are identical. Only 1 of 20 episodes differs from clean.

## 5. Bugs, invalid comparisons and misleading metrics

| ID | Finding | Evidence | Severity |
|---|---|---|---|
| B1 [BUG] | `main()` in the current script prints `NaN` for every poisoned row: it filters `condition == "poisoned"`, but the current `evaluate()` writes `"delta_0.25"`. | Observed on re-execution; stderr "Mean of empty slice" | Medium (reporting only) |
| B2 [BUG] | Provenance mismatch: supplied `results.csv` (6 columns, label "poisoned") can only have been produced by the **backup** version, not the current script, whose output schema is different. | schema comparison + diff | Medium |
| B3 [INVALID COMPARISON] | The shortest-path "poisoned" rows are identical to clean **by construction** (`action=0` ignores observations). They are not evidence of anything. | rows identical for all 20 seeds; code | High for interpretation |
| B4 [MISLEADING] | Environment is saturated: in 98% of episodes (both methods) at least one link exceeds 100% utilization. There is no admission control, packet loss or capacity constraint, and the observation is clipped at 1.0 so the agent cannot see overload. "Max utilization > 1" is demand exceeding capacity, not a delivery failure. | `episodes.csv` | High |
| B5 [MISLEADING] | `mean_utilization = Σ_t volume_t·hops_t / (13·100)`. It is a path-length proxy, not a load-balance measure. | algebra; tests check `total_hops` | Medium |
| B6 [DESIGN] | A single trained model and 20 scenarios were used; training-seed variance (§ report) is larger than the poisoning effect, so "PPO is barely affected" cannot be generalized. | Phase 1 experiment | High |
| B7 [DESIGN] | Poisoning is a fixed offset on two links; link-selection and magnitude are not optimized; the attacker is non-adaptive. | code | Scope limit |
| B8 [DESIGN] | Training reward (negative mean squared utilization, a state signal) is not a security metric. | code | Interpretation |
| B9 [MINOR] | `evaluate_poisoning_levels.py`: unused `metrics` list; hard-coded relative paths; only trajectory-level action change reported (compounds after first divergence). | code | Low |
| B10 [MINOR] | No `requirements.txt`, tests, README, or `.gitignore`; `device="auto"` and default thread count make training machine-dependent. | listing | Low |

## 6. Open questions

* **[OPEN]** Contents and hyperparameters of the original `ppo_routing.zip` (file missing). Whether it was trained by the current script or the backup.
* **[OPEN]** Library versions/hardware of the original run (could explain the PPO reproduction gap in §4, but this cannot be tested without them).
* **[OPEN]** Whether any evaluation scenario coincides with a training episode (not checked; the training demands come from a continuing RNG stream, so exact repeats are unlikely but unverified).
* **[OPEN]** ENERO's original code, topologies and training setup are not available here, so no comparison is possible.
