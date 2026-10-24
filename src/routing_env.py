"""Phase 1 routing environment with optional observation poisoning.

Derived from ``originals/routing_poisoning.py``. The dynamics, RNG consumption,
observation layout, poisoning rule and reward are intentionally IDENTICAL to the
original at default settings (verified by ``tests/test_env_equivalence.py``).

Changes relative to the original (all additive / instrumentation only):
  * Observation construction is split into a clean part and a poisoning part so
    clean and poisoned observations can be compared for the same state.
  * ``counterfactual_observation(delta)`` returns the observation the agent would
    see for the CURRENT state under an arbitrary poisoning strength.
  * ``last_clean_obs`` / ``last_obs`` expose the most recent clean and delivered
    observations, so the actual perturbation can be measured.
  * Terminal ``info`` gains extra metrics (episode_reward, overloaded_links,
    total_hops, ...). The three original keys are unchanged.
  * ``step`` raises ``ValueError`` for out-of-range actions instead of relying on
    an incidental IndexError / negative indexing.
  * Poisoned link indices are a constructor argument (default = original (0, 3)).

The network is a small SYNTHETIC toy topology (8 nodes, 13 undirected links,
uniform capacity). It is not a real-world dataset.
"""

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

N_NODES = 8
EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 3),
    (2, 3), (2, 4), (3, 4), (3, 5),
    (4, 5), (4, 6), (5, 6), (5, 7), (6, 7),
]
CAPACITY = 100.0
N_DEMANDS = 20
K_PATHS = 5
DEFAULT_POISON_EDGES = (0, 3)  # link indices into ``RoutingEnv.edges``
VOLUME_RANGE = (10.0, 35.0)


class RoutingEnv(gym.Env):
    """Sequential traffic routing; one demand is routed per step.

    Observation (float32, length n_edges + 2*N_NODES + 1 = 30), all in [0, 1]:
        [0:13]   per-link utilization, clipped to [0, 1]   (the poisoned part)
        [13:21]  one-hot source node
        [21:29]  one-hot destination node
        [29]     demand volume / CAPACITY
    Action: Discrete(K_PATHS). Index into the K shortest simple paths (by hop
        count) between the current source and destination. Index 0 is the
        shortest-hop path.
    Reward: -mean_e (load_e / CAPACITY)^2, computed from TRUE loads at every step.
    Termination: after N_DEMANDS steps. Never truncated.
    """

    metadata = {"render_modes": []}

    def __init__(self, poison=False, delta=0.25, poison_edges=DEFAULT_POISON_EDGES):
        super().__init__()
        self.poison = poison
        self.delta = float(delta)

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(N_NODES))
        self.graph.add_edges_from(EDGES)

        self.edges = list(self.graph.edges())
        self.n_edges = len(self.edges)
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.poison_edges = tuple(poison_edges)
        for idx in self.poison_edges:
            if not 0 <= idx < self.n_edges:
                raise ValueError(f"poison edge index {idx} out of range")

        obs_size = self.n_edges + 2 * N_NODES + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(K_PATHS)

        self.loads = np.zeros(self.n_edges, dtype=np.float32)
        self.demands = []
        self.step_idx = 0
        self.current_demand = None
        self.paths = []
        self.last_clean_obs = None
        self.last_obs = None
        self._episode_reward = 0.0
        self._total_hops = 0

    # ------------------------------------------------------------------ paths
    def _candidate_paths(self, source, target):
        paths = list(nx.shortest_simple_paths(self.graph, source, target))[:K_PATHS]
        if not paths:
            raise RuntimeError("No path between demand endpoints")
        while len(paths) < K_PATHS:
            paths.append(paths[-1])
        return paths

    def path_edge_indices(self, path):
        """Link indices traversed by a node path."""
        out = []
        for u, v in zip(path[:-1], path[1:]):
            idx = self.edge_index.get((u, v))
            if idx is None:
                idx = self.edge_index[(v, u)]
            out.append(idx)
        return out

    # ------------------------------------------------------------ observation
    def _clean_observation(self):
        source, target, volume = self.current_demand
        true_util = self.loads / CAPACITY
        observed = np.clip(true_util, 0.0, 1.0).copy()
        src = np.zeros(N_NODES, dtype=np.float32)
        dst = np.zeros(N_NODES, dtype=np.float32)
        src[source] = 1.0
        dst[target] = 1.0
        return np.concatenate([
            observed.astype(np.float32),
            src,
            dst,
            np.array([volume / CAPACITY], dtype=np.float32),
        ])

    def _apply_poison(self, clean_obs, delta):
        """Additive, clipped falsification of the poisoned links' utilization.

        Only observation entries are changed; simulator loads are untouched.
        With delta == 0 this returns an exact copy of ``clean_obs``.
        """
        obs = clean_obs.copy()
        for idx in self.poison_edges:
            obs[idx] = np.clip(obs[idx] + np.float32(delta), 0.0, 1.0)
        return obs

    def _observation(self):
        clean = self._clean_observation()
        obs = self._apply_poison(clean, self.delta) if self.poison else clean
        self.last_clean_obs = clean
        self.last_obs = obs
        return obs

    def counterfactual_observation(self, delta):
        """Observation for the CURRENT state if poisoned with ``delta``."""
        return self._apply_poison(self._clean_observation(), delta)

    # ---------------------------------------------------------------- dynamics
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.loads[:] = 0.0
        self.step_idx = 0
        self.demands = []
        self._episode_reward = 0.0
        self._total_hops = 0

        for _ in range(N_DEMANDS):
            source, target = self.np_random.choice(N_NODES, size=2, replace=False)
            volume = float(self.np_random.uniform(*VOLUME_RANGE))
            self.demands.append((int(source), int(target), volume))

        self.current_demand = self.demands[0]
        s, t, _ = self.current_demand
        self.paths = self._candidate_paths(s, t)
        return self._observation(), {}

    def step(self, action):
        action = int(action)
        if not 0 <= action < K_PATHS:
            raise ValueError(f"invalid action {action}; expected 0..{K_PATHS - 1}")
        path = self.paths[action]
        _, _, volume = self.current_demand

        # Traffic is applied to the TRUE network state.
        for idx in self.path_edge_indices(path):
            self.loads[idx] += volume
        self._total_hops += len(path) - 1

        true_util = self.loads / CAPACITY
        reward = -float(np.mean(true_util ** 2))  # ground truth, never poisoned
        self._episode_reward += reward

        self.step_idx += 1
        terminated = self.step_idx >= N_DEMANDS
        truncated = False

        if not terminated:
            self.current_demand = self.demands[self.step_idx]
            s, t, _ = self.current_demand
            self.paths = self._candidate_paths(s, t)
            obs = self._observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {}
        if terminated:
            info = {
                # original keys (unchanged definitions)
                "max_utilization": float(np.max(true_util)),
                "mean_utilization": float(np.mean(true_util)),
                "squared_utilization": float(np.mean(true_util ** 2)),
                # Phase 1 additions
                "episode_reward": float(self._episode_reward),
                "overloaded_links": int(np.sum(true_util > 1.0)),
                "total_hops": int(self._total_hops),
            }
        return obs, reward, terminated, truncated, info
