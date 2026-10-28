"""Flow-level, capacity-aware network simulator (no packet queues).

TRUE STATE : per-directed-edge accepted load (capacity units, CU = kbps in the NED).
MODEL      : demands arrive one at a time; each is routed on ONE path chosen by the policy.
ADMISSION  : accepted = min(volume, min_e residual_e on the path), residual_e = cap_e - load_e.
             So load_e <= cap_e ALWAYS (utilization never exceeds 1). The rest is BLOCKED.
             There is no packet-level loss, queueing or delay model: "blocked volume" is an
             admission-control quantity, not packet loss, and delay is NOT modelled.
Flows persist for the rest of the episode (a static traffic-matrix-style scenario).
"""
import heapq
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .topology import Topology, load_ned

ROOT = Path(__file__).resolve().parents[2]
NED = ROOT / "data/raw/nsfnetbw_fork/Network_nsfnetbw.ned"

N_FLOWS = 60                    # demands per episode
REF_TOTAL_RATE = 69.326         # CU: median total offered rate of the 500 shipped lambda=9 samples
U_RANGE = (0.2, 1.8)            # per-flow volume multiplier, mean 1
K_PATHS = 4
EXTRA_HOPS = 2                  # initial hop slack; widened per pair until K simple paths exist, then the K best by (hops, lexicographic) are kept
VOL_NORM = 10.0                 # = smallest link capacity; observation volume feature = vol/VOL_NORM
TRAIN_SEED_RANGE = (0, 1_000_000)
VAL_SEED_BASE = 10_000_000
TEST_SEED_BASE = 20_000_000


@dataclass
class Scenario:
    seed: int
    load: float
    src: np.ndarray
    dst: np.ndarray
    vol: np.ndarray


def make_scenario(seed, load, n_nodes=14, n_flows=N_FLOWS):
    """Deterministic in (seed, load). Pair draws depend ONLY on seed; volumes = base_u * load."""
    rng = np.random.default_rng(seed)
    src = np.empty(n_flows, dtype=np.int64); dst = np.empty(n_flows, dtype=np.int64)
    for i in range(n_flows):
        s, d = rng.choice(n_nodes, size=2, replace=False); src[i], dst[i] = s, d
    u = rng.uniform(*U_RANGE, size=n_flows)
    vol = load * REF_TOTAL_RATE / n_flows * u
    return Scenario(int(seed), float(load), src, dst, vol)


class Network:
    def __init__(self, topo=None):
        self.topo = topo or load_ned(NED)
        self.n = self.topo.n_nodes
        de = self.topo.directed_edges()
        self.E = len(de)
        self.cap = np.array([c for *_, c in de], dtype=np.float64)
        self.eid = {(u, v): i for i, (u, v, _) in enumerate(de)}
        self.adj = [[] for _ in range(self.n)]           # (neighbour, edge_id)
        for (u, v), i in self.eid.items():
            self.adj[u].append((v, i))
        for a in self.adj: a.sort()
        G = nx.Graph(); G.add_edges_from((u, v) for u, v, *_ in self.topo.undirected)
        self.G = G
        self._cand = {}
        for s in range(self.n):
            for d in range(self.n):
                if s == d: continue
                sp = nx.shortest_path_length(G, s, d)
                extra = EXTRA_HOPS
                while True:   # smallest hop slack that yields >= K simple paths
                    ps = list(nx.all_simple_paths(G, s, d, cutoff=sp + extra))
                    if len(ps) >= K_PATHS or extra > 12: break
                    extra += 1
                ps.sort(key=lambda p: (len(p), p))
                ps = ps[:K_PATHS]
                if len(ps) < K_PATHS:
                    raise RuntimeError(f"pair {(s, d)} has only {len(ps)} candidate paths")
                self._cand[(s, d)] = [(tuple(p), np.array([self.eid[(a, b)] for a, b in zip(p[:-1], p[1:])])) for p in ps]

    @property
    def max_candidate_hops(self):
        return max(len(e) for c in self._cand.values() for _, e in c)

    def candidates(self, s, d):
        return self._cand[(s, d)]

    # ---- routing primitives (weights/residuals are arrays over directed edges)
    def dijkstra(self, s, d, w):
        dist = [np.inf] * self.n; prev = [None] * self.n; dist[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > dist[u]: continue
            if u == d: break
            for v, i in self.adj[u]:
                nd = du + w[i]
                if nd < dist[v] - 1e-12:
                    dist[v] = nd; prev[v] = (u, i); heapq.heappush(pq, (nd, v))
        edges, node = [], d
        while node != s:
            u, i = prev[node]; edges.append(i); node = u
        edges.reverse()
        return np.array(edges, dtype=np.int64)

    def feasible_min_hop(self, s, d, residual, vol):
        """BFS min-hop path using only edges with residual >= vol (neighbours in sorted order)."""
        prev = {s: None}; q = [s]
        for u in q:
            if u == d: break
            for v, i in self.adj[u]:
                if v not in prev and residual[i] + 1e-12 >= vol:
                    prev[v] = (u, i); q.append(v)
        if d not in prev: return None
        edges, node = [], d
        while node != s:
            u, i = prev[node]; edges.append(i); node = u
        return np.array(edges[::-1], dtype=np.int64)

    def widest_path(self, s, d, residual):
        """Max-bottleneck-residual path; ties broken by fewer hops then lexicographic order."""
        best = {s: (np.inf, 0)}; prev = {s: None}; pq = [(-np.inf, 0, s)]
        done = set()
        while pq:
            nb, h, u = heapq.heappop(pq)
            if u in done: continue
            done.add(u)
            if u == d: break
            for v, i in self.adj[u]:
                b = min(-nb, residual[i]); cand = (b, -(h + 1))
                if v not in best or cand > (best[v][0], -best[v][1]):
                    best[v] = (b, h + 1); prev[v] = (u, i); heapq.heappush(pq, (-b, h + 1, v))
        edges, node = [], d
        while node != s:
            u, i = prev[node]; edges.append(i); node = u
        return np.array(edges[::-1], dtype=np.int64)


class Episode:
    """Mutable true state for one scenario. Policies never see `loads` directly, only telemetry."""
    def __init__(self, net, scen):
        self.net, self.scen = net, scen
        self.loads = np.zeros(net.E); self.t = 0
        self.total_offered = float(scen.vol.sum())
        self.accepted_total = 0.0; self.hop_volume = 0.0; self.n_full_block = 0; self.n_partial = 0
        self.n_hops_total = 0; self.n_throttled = 0

    @property
    def done(self): return self.t >= len(self.scen.vol)
    def true_util(self): return self.loads / self.net.cap
    def residual(self): return self.net.cap - self.loads
    def demand(self):
        i = self.t; return int(self.scen.src[i]), int(self.scen.dst[i]), float(self.scen.vol[i])

    def route(self, edges):
        """Apply admission control on path `edges`; returns accepted volume."""
        _, _, vol = self.demand()
        edges = np.asarray(edges, dtype=np.int64)
        acc = float(max(0.0, min(vol, float(np.min(self.net.cap[edges] - self.loads[edges])))))
        self.loads[edges] += acc
        self.loads = np.minimum(self.loads, self.net.cap)       # guard against float round-off only
        self.accepted_total += acc; self.hop_volume += acc * len(edges); self.n_hops_total += len(edges)
        if acc <= 1e-9: self.n_full_block += 1
        elif acc < vol - 1e-9: self.n_partial += 1
        self.t += 1
        return acc

    def metrics(self):
        u = self.true_util(); n = len(self.scen.vol)
        acc = self.accepted_total
        return dict(
            offered=self.total_offered, delivered=acc,
            delivery_ratio=acc / self.total_offered, blocking_ratio=1.0 - acc / self.total_offered,
            flows_fully_blocked=self.n_full_block / n, flows_partially_admitted=self.n_partial / n,
            max_utilization=float(u.max()), mean_utilization=float(u.mean()),
            saturated_links=int((u >= 0.999).sum()),
            mean_hops_accepted=self.hop_volume / acc if acc > 0 else float("nan"),
            capacity_hops_used=self.hop_volume, mean_hops_per_flow=self.n_hops_total / n,
        )
