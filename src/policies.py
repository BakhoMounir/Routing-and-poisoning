"""Non-learning reference policies (Phase 1).

Both take ``(env, obs)`` and return an action index into ``env.paths``.

shortest_path_action
    Static shortest-hop routing: always candidate index 0 (the first path from
    ``networkx.shortest_simple_paths``, i.e. a minimum-hop path). It uses neither
    link loads nor telemetry, so observation poisoning CANNOT affect it. This is
    exactly what the original code did (``action = 0``). Its "poisoned" results
    are identical to its clean results by construction, not by measurement.

greedy_telemetry_action
    DIAGNOSTIC reference added in Phase 1 (not in the original code, not an RL
    method). It reads the SAME observation the PPO agent receives and picks the
    candidate path that minimises, lexicographically, (max observed utilization
    on the path after adding this demand, sum of observed utilization, hops).
    Because it consumes telemetry, it shows whether a perturbation of the
    observation can change decisions of a policy that actually uses the
    poisoned features. It inherits the observation's clipping at 1.0, so it is
    blind to overload beyond 100% exactly like the PPO agent.
"""

from .routing_env import CAPACITY


def shortest_path_action(env, obs):  # noqa: ARG001 - signature shared by policies
    return 0


def greedy_telemetry_action(env, obs):
    n = env.n_edges
    util = obs[:n]
    demand_frac = float(obs[-1])  # volume / CAPACITY
    best_key, best_action = None, 0
    for k, path in enumerate(env.paths):
        idx = env.path_edge_indices(path)
        loads_after = [float(util[i]) + demand_frac for i in idx]
        key = (round(max(loads_after), 6), round(sum(loads_after), 6), len(idx), k)
        if best_key is None or key < best_key:
            best_key, best_action = key, k
    return best_action


__all__ = ["shortest_path_action", "greedy_telemetry_action", "CAPACITY"]
