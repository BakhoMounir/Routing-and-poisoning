"""Static / greedy baselines. Each takes (net, episode, obs_util) and returns directed-edge indices.
`obs_util` is the OBSERVED telemetry (clean == true utilization now); only min-hop ignores it."""
import numpy as np


def minhop(net, ep, obs_util):
    """Static min-hop shortest path = candidate 0 (fewest hops, lexicographic tie-break). Ignores telemetry."""
    s, d, _ = ep.demand()
    return net.candidates(s, d)[0][1]


def shortest_feasible(net, ep, obs_util):
    """Capacity-aware: min-hop path over edges whose OBSERVED residual >= demand; if none, widest path."""
    s, d, vol = ep.demand()
    resid = (1.0 - obs_util) * net.cap
    p = net.feasible_min_hop(s, d, resid, vol)
    return p if p is not None else net.widest_path(s, d, resid)


def greedy_widest_candidate(net, ep, obs_util):
    """Telemetry-greedy over the SAME K candidates the replacement agent sees: max bottleneck residual, then fewer hops."""
    s, d, _ = ep.demand()
    resid = (1.0 - obs_util) * net.cap
    best, key = None, None
    for k, (_, e) in enumerate(net.candidates(s, d)):
        kk = (-round(float(resid[e].min()), 9), len(e), k)
        if key is None or kk < key: key, best = kk, e
    return best


BASELINES = {"minhop": minhop, "shortest_feasible": shortest_feasible, "greedy_widest": greedy_widest_candidate}
