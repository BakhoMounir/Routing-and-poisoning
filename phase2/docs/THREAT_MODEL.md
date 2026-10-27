# Threat model (predeclared for Stage F)

**Goal.** Test whether additive corruption of the utilization values *observed by the routing policy at test time* degrades routing, without touching anything else.

**Unchanged by construction (and tested):** true link loads, capacities, traffic, admission, reward, trained parameters. Only `obs_util` handed to the policy is perturbed:
`obs = clip(true + delta, 0, 1)`, `|delta_e| <= eps` for every directed edge, applied at every decision.

**Attacker knowledge.** `random_sign`, `bias_all_*`: none (oblivious). `conceal_topk` / `fake_topk`: attacker sees *true* utilization to pick edges (white-box on telemetry, black-box on the model; no gradients, non-adaptive, no query feedback).
**Not modelled:** adaptive/gradient attacks, perturbations of demand/src/dst features, attacks at training time, attacker cost, detection. Results are therefore *lower bounds* on vulnerability to stronger attackers, and say nothing about real networks.

| Attack | Effect |
|---|---|
| random_sign | +-eps i.i.d.; **noise control** at equal budget |
| bias_all_plus / minus | uniform over/under-reporting |
| conceal_topk | -eps on the k=8 most-utilized edges (hide congestion) |
| fake_topk | +eps on the k=8 least-utilized edges (invent congestion) |
eps grid {0.1, 0.2, 0.4} (pilot {0.2, 0.4}). Clean control (`clean`) is run on identical scenarios and models.

**Controls.** `minhop`, `unit_weight_dijkstra` ignore telemetry (must be exactly unaffected - asserted); `shortest_feasible`, `greedy_widest`, `shortest_feasible_k`, `util_weight_dijkstra` read telemetry (non-learned references).
**Endpoints.** accepted_offered_ratio change vs clean (same model, same scenario, pp); decision flip rate (counterfactual: same state, clean vs perturbed observation).
**Primary tests (Holm over 4 cells):** targets A_600k and B_legacy_150k x {conceal_topk, fake_topk} at eps 0.2, load 2.0: attacked-minus-random_sign contrast. **"Attack success" is claimed only if** the crossed CI of (attacked - clean) excludes 0 *and* the contrast vs random_sign excludes 0 after Holm. Everything else is exploratory.
Telemetry sensitivity (e.g. the zero-telemetry ablation) is **not** evidence of vulnerability.
