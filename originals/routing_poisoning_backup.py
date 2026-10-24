import csv
import random
from pathlib import Path

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO


# Explicit toy topology for testing the experimental pipeline.
# Replace with a verified research topology before reporting results.
N_NODES = 8
EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 3),
    (2, 3), (2, 4), (3, 4), (3, 5),
    (4, 5), (4, 6), (5, 6), (5, 7), (6, 7),
]
CAPACITY = 100.0
N_DEMANDS = 20
K_PATHS = 5


class RoutingEnv(gym.Env):
    """Sequential traffic routing with optional observation poisoning."""

    metadata = {"render_modes": []}

    def __init__(self, poison=False, delta=0.25):
        super().__init__()
        self.poison = poison
        self.delta = delta

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(N_NODES))
        self.graph.add_edges_from(EDGES)

        self.edges = list(self.graph.edges())
        self.n_edges = len(self.edges)
        self.edge_index = {e: i for i, e in enumerate(self.edges)}

        # Observation: edge utilizations, source/destination one-hot,
        # and normalized demand volume.
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

    def _candidate_paths(self, source, target):
        paths = list(
            nx.shortest_simple_paths(self.graph, source, target)
        )[:K_PATHS]
        if not paths:
            raise RuntimeError("No path between demand endpoints")
        while len(paths) < K_PATHS:
            paths.append(paths[-1])
        return paths

    def _observation(self):
        source, target, volume = self.current_demand
        true_util = self.loads / CAPACITY
        observed = np.clip(true_util, 0.0, 1.0).copy()

        if self.poison:
            # Falsify telemetry for two fixed links only.
            # The simulator's actual loads remain unchanged.
            for idx in (0, min(3, self.n_edges - 1)):
                observed[idx] = np.clip(
                    observed[idx] + self.delta, 0.0, 1.0
                )

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.loads[:] = 0.0
        self.step_idx = 0
        self.demands = []

        for _ in range(N_DEMANDS):
            source, target = self.np_random.choice(
                N_NODES, size=2, replace=False
            )
            volume = float(self.np_random.uniform(10.0, 35.0))
            self.demands.append((int(source), int(target), volume))

        self.current_demand = self.demands[0]
        s, t, _ = self.current_demand
        self.paths = self._candidate_paths(s, t)
        return self._observation(), {}

    def step(self, action):
        path = self.paths[int(action)]
        _, _, volume = self.current_demand

        # Apply traffic to the TRUE network state.
        for u, v in zip(path[:-1], path[1:]):
            idx = self.edge_index.get((u, v))
            if idx is None:
                idx = self.edge_index[(v, u)]
            self.loads[idx] += volume

        true_util = self.loads / CAPACITY

        # Congestion penalty based on ground truth, not poisoned telemetry.
        reward = -float(np.mean(true_util ** 2))

        self.step_idx += 1
        terminated = self.step_idx >= N_DEMANDS
        truncated = False

        if not terminated:
            self.current_demand = self.demands[self.step_idx]
            s, t, _ = self.current_demand
            self.paths = self._candidate_paths(s, t)
            obs = self._observation()
        else:
            obs = np.zeros(
                self.observation_space.shape, dtype=np.float32
            )

        info = {}
        if terminated:
            info = {
                "max_utilization": float(np.max(true_util)),
                "mean_utilization": float(np.mean(true_util)),
                "squared_utilization": float(np.mean(true_util ** 2)),
            }

        return obs, reward, terminated, truncated, info


def evaluate(model, poison, seeds=range(100, 120), baseline=False):
    rows = []
    env = RoutingEnv(poison=poison)

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        final_info = {}

        while not done:
            if baseline:
                # Conventional unweighted shortest-hop path.
                action = 0
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                final_info = info

        rows.append({
            "seed": seed,
            "method": "shortest_path" if baseline else "ppo",
            "condition": "poisoned" if poison else "clean",
            **final_info,
        })

    env.close()
    return rows


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Training PPO on clean observations...")
    train_env = RoutingEnv(poison=False)

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=42,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        verbose=1,
        device="auto",
    )
    model.learn(total_timesteps=20_000)
    model.save("ppo_routing")
    train_env.close()

    rows = []
    rows += evaluate(model, poison=False)
    rows += evaluate(model, poison=True)
    rows += evaluate(model, poison=False, baseline=True)
    rows += evaluate(model, poison=True, baseline=True)

    output = Path("results.csv")
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} episode results to {output}")
    for condition in ("clean", "poisoned"):
        for method in ("ppo", "shortest_path"):
            subset = [
                r for r in rows
                if r["condition"] == condition and r["method"] == method
            ]
            print(
                f"{method:14s} {condition:9s} | "
                f"mean max utilization = "
                f"{np.mean([r['max_utilization'] for r in subset]):.4f} | "
                f"mean link utilization = "
                f"{np.mean([r['mean_utilization'] for r in subset]):.4f}"
            )


if __name__ == "__main__":
    main()
