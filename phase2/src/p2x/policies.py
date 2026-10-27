"""Additional NON-LEARNED reference policies. Same signature as p2.baselines: fn(net, ep, obs_util) -> edge indices.

They read ONLY the observed utilization `obs_util` and the current demand (ep.demand()), never ep.loads,
so they can be run under a perturbed telemetry channel as controls. None of these is an RL model.
"""
from __future__ import annotations

import numpy as np

from p2.baselines import BASELINES
from p2.envs import EnhanceEnv

FEAS_TOL = 1e-12            # same tolerance as Network.feasible_min_hop
W_MIN, W_MAX = 0.1, 10.0    # EnhanceEnv weight range: w = exp(a*ln10), a in [-1, 1]


def shortest_feasible_k(net, ep, obs_util):
    """shortest_feasible RESTRICTED to Agent A's K candidate paths (same action set as Agent A).
    First candidate (fewest hops, lexicographic) whose observed bottleneck residual >= demand; otherwise the
    widest candidate (max bottleneck, then fewer hops). Isolates the effect of the K-candidate restriction."""
    s, d, vol = ep.demand()
    resid = (1.0 - obs_util) * net.cap
    cands = net.candidates(s, d)
    for _, e in cands:
        if float(resid[e].min()) + FEAS_TOL >= vol:
            return e
    best, key = None, None
    for k, (_, e) in enumerate(cands):
        kk = (-round(float(resid[e].min()), 9), len(e), k)
        if key is None or kk < key:
            key, best = kk, e
    return best


def unit_weight_dijkstra(net, ep, obs_util):
    """Agent B's null: Dijkstra with all weights = 1 (what a zero action produces). Ignores telemetry."""
    s, d, _ = ep.demand()
    return net.dijkstra(s, d, np.ones(net.E))


class UtilWeightDijkstra:
    """Transparent heuristic routed through the SAME pipeline as Agent B (action in [-1,1]^E -> EnhanceEnv.weights -> Dijkstra).
    w_e = (1 / max(1 - u_e, floor)) ** p  [x infeasible_multiplier if observed residual_e < demand], clipped to [0.1, 10]."""

    def __init__(self, p: float, floor: float = 0.1, infeasible_multiplier: float = 1.0):
        if p < 0 or not (0 < floor <= 1) or infeasible_multiplier < 1:
            raise ValueError(f"invalid heuristic parameters p={p} floor={floor} mult={infeasible_multiplier}")
        self.p, self.floor, self.mult = float(p), float(floor), float(infeasible_multiplier)

    def action(self, net, obs_util, vol) -> np.ndarray:
        u = np.clip(np.asarray(obs_util, dtype=np.float64), 0.0, 1.0)
        w = (1.0 / np.maximum(1.0 - u, self.floor)) ** self.p
        if self.mult != 1.0:
            infeasible = (1.0 - u) * net.cap + FEAS_TOL < vol
            w = w * np.where(infeasible, self.mult, 1.0)
        return np.log10(np.clip(w, W_MIN, W_MAX))          # exactly the policy action space [-1, 1]^E

    def __call__(self, net, ep, obs_util):
        s, d, vol = ep.demand()
        return net.dijkstra(s, d, EnhanceEnv.weights(self.action(net, obs_util, vol)))

    def describe(self) -> dict:
        return {"p": self.p, "floor": self.floor, "infeasible_multiplier": self.mult}


def build_baselines(heuristic_params: dict | None = None) -> dict:
    """All non-learned methods by id. `util_weight_dijkstra` is present only when its parameters are given."""
    out = dict(BASELINES)
    out["shortest_feasible_k"] = shortest_feasible_k
    out["unit_weight_dijkstra"] = unit_weight_dijkstra
    if heuristic_params is not None:
        out["util_weight_dijkstra"] = UtilWeightDijkstra(**heuristic_params)
    return out


def heuristic_grid(cfg: dict):
    g = cfg["heuristic_grid"]
    return [dict(p=p, floor=g["floor"], infeasible_multiplier=m) for p in g["p"] for m in g["infeasible_multiplier"]]
