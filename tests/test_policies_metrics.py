"""Baselines and metric definitions."""

import os
from pathlib import Path

import numpy as np
import pytest

from src.policies import greedy_telemetry_action, shortest_path_action
from src.routing_env import CAPACITY, K_PATHS, N_DEMANDS, RoutingEnv

MODELS = Path(__file__).resolve().parents[1] / "results" / "models"


def run(env, fn, seed):
    obs, _ = env.reset(seed=seed)
    acts, done, info = [], False, {}
    while not done:
        a = fn(env, obs)
        acts.append(a)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return acts, info


def test_shortest_path_is_independent_of_poisoning():
    for seed in range(100, 120):
        a_c, i_c = run(RoutingEnv(poison=False), shortest_path_action, seed)
        a_p, i_p = run(RoutingEnv(poison=True, delta=1.0), shortest_path_action, seed)
        assert a_c == a_p and i_c == i_p


def test_shortest_path_candidate_zero_is_a_minimum_hop_path():
    env = RoutingEnv()
    env.reset(seed=100)
    for _ in range(N_DEMANDS):
        hops = [len(p) - 1 for p in env.paths]
        assert hops[0] == min(hops) and hops == sorted(hops)
        env.step(0)


def test_greedy_telemetry_baseline_does_read_the_poisoned_observation():
    differing = 0
    for seed in range(100, 120):
        a_c, _ = run(RoutingEnv(poison=False), greedy_telemetry_action, seed)
        a_p, _ = run(RoutingEnv(poison=True, delta=1.0), greedy_telemetry_action, seed)
        differing += int(a_c != a_p)
    assert differing > 0


def test_metric_definitions_are_consistent_with_true_state():
    env = RoutingEnv()
    env.reset(seed=105)
    rewards, hops = [], 0
    for a in np.random.default_rng(0).integers(0, K_PATHS, size=N_DEMANDS):
        hops += len(env.paths[a]) - 1
        _, r, term, _, info = env.step(a)
        rewards.append(r)
    util = env.loads / CAPACITY
    assert info["max_utilization"] == pytest.approx(float(util.max()))
    assert info["mean_utilization"] == pytest.approx(float(util.mean()))
    assert info["squared_utilization"] == pytest.approx(float((util ** 2).mean()))
    assert info["episode_reward"] == pytest.approx(sum(rewards))
    assert rewards[-1] == pytest.approx(-info["squared_utilization"])
    assert info["overloaded_links"] == int((util > 1.0).sum())
    assert info["total_hops"] == hops
    assert info["max_utilization"] >= info["mean_utilization"] >= 0.0


@pytest.mark.skipif(not (MODELS / "ppo_seed42.zip").exists(), reason="run scripts/train.py first")
def test_trained_model_is_deterministic_and_actions_valid():
    from stable_baselines3 import PPO

    m1 = PPO.load(str(MODELS / "ppo_seed42.zip"), device="cpu")
    m2 = PPO.load(str(MODELS / "ppo_seed42.zip"), device="cpu")
    f = lambda m: (lambda env, obs: int(m.predict(obs, deterministic=True)[0]))  # noqa: E731
    for seed in range(100, 110):
        a1, i1 = run(RoutingEnv(), f(m1), seed)
        a2, i2 = run(RoutingEnv(), f(m2), seed)
        assert a1 == a2 and i1 == i2 and all(0 <= a < K_PATHS for a in a1)


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow; set RUN_SLOW=1")
def test_training_is_bitwise_repeatable_on_cpu_single_thread(tmp_path):
    import hashlib
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train

    hashes = []
    for i in range(2):
        train.train_one(7, tmp_path / f"run{i}")
        hashes.append(json.loads((tmp_path / f"run{i}" / "ppo_seed7.json").read_text())["param_sha256"])
    assert hashes[0] == hashes[1]
    assert hashlib.sha256  # silence linters
