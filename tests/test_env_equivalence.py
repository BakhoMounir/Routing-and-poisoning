"""Phase 1 environment must behave exactly like the preserved original."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.policies import shortest_path_action
from src.routing_env import K_PATHS, N_DEMANDS, RoutingEnv

ORIG = Path(__file__).resolve().parents[1] / "originals"


@pytest.fixture(scope="module")
def orig():
    spec = importlib.util.spec_from_file_location("orig_routing_poisoning", ORIG / "routing_poisoning.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("poison,delta", [(False, 0.25), (True, 0.1), (True, 0.25), (True, 0.5)])
def test_observations_rewards_and_info_identical_to_original(orig, poison, delta):
    for seed in range(100, 110):
        e_new = RoutingEnv(poison=poison, delta=delta)
        e_old = orig.RoutingEnv(poison=poison, delta=delta)
        o_n, _ = e_new.reset(seed=seed)
        o_o, _ = e_old.reset(seed=seed)
        assert e_new.demands == e_old.demands and np.array_equal(o_n, o_o)
        acts = np.random.default_rng(seed).integers(0, K_PATHS, size=N_DEMANDS)
        for a in acts:
            o_n, r_n, t_n, _, i_n = e_new.step(a)
            o_o, r_o, t_o, _, i_o = e_old.step(a)
            assert np.array_equal(o_n, o_o) and r_n == r_o and t_n == t_o
            for k, v in i_o.items():  # original keys must be unchanged
                assert i_n[k] == v


def test_shortest_path_reproduces_original_results_csv():
    df = pd.read_csv(ORIG / "results.csv")
    df = df[(df.method == "shortest_path") & (df.condition == "clean")].set_index("seed")
    assert len(df) == 20
    for seed in range(100, 120):
        env = RoutingEnv(poison=False)
        obs, _ = env.reset(seed=seed)
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(shortest_path_action(env, obs))
            done = term or trunc
        for k in ("max_utilization", "mean_utilization", "squared_utilization"):
            assert info[k] == pytest.approx(df.loc[seed, k], rel=1e-6)


def test_original_csv_internal_consistency():
    """Facts about the preserved original CSVs that the report relies on."""
    r = pd.read_csv(ORIG / "results.csv")
    s = pd.read_csv(ORIG / "poisoning_strength_results.csv")
    cols = ["max_utilization", "mean_utilization", "squared_utilization"]
    sp_c = r[(r.method == "shortest_path") & (r.condition == "clean")][cols].to_numpy()
    sp_p = r[(r.method == "shortest_path") & (r.condition == "poisoned")][cols].to_numpy()
    assert np.array_equal(sp_c, sp_p)  # SP ignores observations
    ppo = r[r.method == "ppo"].groupby("condition")[cols].mean()
    assert ppo.loc["clean", "max_utilization"] == pytest.approx(1.5082, abs=5e-5)
    assert ppo.loc["clean", "mean_utilization"] == pytest.approx(0.6550, abs=5e-5)
    assert ppo.loc["poisoned", "mean_utilization"] == pytest.approx(0.6557, abs=5e-5)
    sp = r[r.method == "shortest_path"].groupby("condition")[cols].mean()
    assert sp.loc["clean", "max_utilization"] == pytest.approx(1.5040, abs=5e-5)
    assert sp.loc["clean", "mean_utilization"] == pytest.approx(0.6237, abs=5e-5)
    clean = s[s.delta == 0.0].set_index("seed")
    for d in (0.1, 0.25, 0.5):
        att = s[s.delta == d].set_index("seed")
        ch = sum(a != b for sd in clean.index
                 for a, b in zip(json.loads(clean.loc[sd, "actions"]), json.loads(att.loc[sd, "actions"])))
        assert ch == (0 if d == 0.1 else 1)
