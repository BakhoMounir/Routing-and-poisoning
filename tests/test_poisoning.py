"""Task B: does poisoning really change what the agent sees, and only that?"""

import numpy as np
import pytest

from src.routing_env import K_PATHS, N_DEMANDS, RoutingEnv

DELTAS = [0.1, 0.25, 0.5, 1.0]


def fixed_actions(seed, n=N_DEMANDS):
    return np.random.default_rng(seed).integers(0, K_PATHS, size=n).tolist()


def trace(env, seed, actions):
    """Return per-step (delivered_obs, clean_obs) plus rewards/loads/final info."""
    obs, _ = env.reset(seed=seed)
    seen, clean, rewards, loads, info = [obs.copy()], [env.last_clean_obs.copy()], [], [], {}
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        loads.append(env.loads.copy())
        if not (term or trunc):
            seen.append(obs.copy())
            clean.append(env.last_clean_obs.copy())
        if term or trunc:
            break
    return seen, clean, rewards, loads, info


def test_zero_delta_poison_reproduces_clean_exactly():
    for seed in range(100, 110):
        acts = fixed_actions(seed)
        s_p, _, r_p, _, i_p = trace(RoutingEnv(poison=True, delta=0.0), seed, acts)
        s_c, _, r_c, _, i_c = trace(RoutingEnv(poison=False), seed, acts)
        assert all(np.array_equal(a, b) for a, b in zip(s_p, s_c))
        assert r_p == r_c and i_p == i_c


@pytest.mark.parametrize("delta", DELTAS)
def test_only_targeted_entries_change(delta):
    env = RoutingEnv(poison=True, delta=delta)
    targeted = set(env.poison_edges)
    changed_somewhere = False
    for seed in range(100, 110):
        seen, clean, *_ = trace(env, seed, fixed_actions(seed))
        for o, c in zip(seen, clean):
            idx = set(np.flatnonzero(o != c).tolist())
            assert idx <= targeted, f"non-target entries changed: {idx - targeted}"
            changed_somewhere |= bool(idx)
    assert changed_somewhere, "poisoning never changed any observation"


@pytest.mark.parametrize("delta", DELTAS)
def test_actual_perturbation_matches_configured_strength(delta):
    """|obs_p - obs_c| == min(delta, 1 - clean) on poisoned links (clipping)."""
    env = RoutingEnv(poison=True, delta=delta)
    for seed in range(100, 105):
        seen, clean, *_ = trace(env, seed, fixed_actions(seed))
        for o, c in zip(seen, clean):
            for idx in env.poison_edges:
                expected = min(delta, 1.0 - float(c[idx]))
                assert abs(float(o[idx] - c[idx]) - expected) < 1e-6
                assert o[idx] - c[idx] >= 0.0  # attack only ever inflates utilization


def test_saturated_link_makes_poison_a_noop():
    env = RoutingEnv(poison=True, delta=0.5)
    env.reset(seed=100)
    env.loads[list(env.poison_edges)] = 250.0  # utilization 2.5 -> clipped to 1.0
    clean = env._clean_observation()
    poisoned = env.counterfactual_observation(0.5)
    assert np.array_equal(clean, poisoned)
    assert all(clean[i] == 1.0 for i in env.poison_edges)


@pytest.mark.parametrize("delta", [-0.3, 0.0, 0.1, 0.25, 0.5, 1.0, 5.0])
def test_observation_bounds_dtype_and_finiteness(delta):
    env = RoutingEnv(poison=True, delta=delta)
    for seed in range(100, 105):
        seen, *_ = trace(env, seed, fixed_actions(seed))
        for o in seen:
            assert o.dtype == np.float32 and o.shape == env.observation_space.shape
            assert np.all(np.isfinite(o)) and o.min() >= 0.0 and o.max() <= 1.0
            assert env.observation_space.contains(o)


def test_poisoning_does_not_alter_true_network_state_or_reward():
    for seed in range(100, 110):
        acts = fixed_actions(seed)
        _, _, r_c, l_c, i_c = trace(RoutingEnv(poison=False), seed, acts)
        _, _, r_p, l_p, i_p = trace(RoutingEnv(poison=True, delta=0.5), seed, acts)
        assert r_c == r_p and i_c == i_p
        assert all(np.array_equal(a, b) for a, b in zip(l_c, l_p))


def test_reproducible_under_fixed_seed_and_distinct_across_seeds():
    acts = fixed_actions(7)
    a = trace(RoutingEnv(poison=True, delta=0.25), 123, acts)
    b = trace(RoutingEnv(poison=True, delta=0.25), 123, acts)
    assert all(np.array_equal(x, y) for x, y in zip(a[0], b[0])) and a[4] == b[4]
    c = trace(RoutingEnv(poison=True, delta=0.25), 124, acts)
    assert not all(np.array_equal(x, y) for x, y in zip(a[0], c[0]))


@pytest.mark.parametrize("delta", DELTAS)
def test_counterfactual_observation_matches_poisoned_environment(delta):
    clean_env, pois_env = RoutingEnv(poison=False), RoutingEnv(poison=True, delta=delta)
    o_c, _ = clean_env.reset(seed=101)
    o_p, _ = pois_env.reset(seed=101)
    for a in fixed_actions(101)[:-1]:
        assert np.array_equal(clean_env.counterfactual_observation(delta), o_p)
        o_c, *_ = clean_env.step(a)
        o_p, *_ = pois_env.step(a)


def test_invalid_configuration_and_actions_fail_loudly():
    with pytest.raises(ValueError):
        RoutingEnv(poison=True, poison_edges=(0, 99))
    env = RoutingEnv()
    env.reset(seed=100)
    for bad in (-1, K_PATHS):
        with pytest.raises(ValueError):
            env.step(bad)


class _Stub:
    """Minimal stand-in for an SB3 model (predict API only)."""

    def __init__(self, fn):
        self.fn = fn

    def predict(self, obs, deterministic=True):
        return np.array(self.fn(obs)), None


def test_flip_measurement_pipeline_detects_sensitive_policy_and_not_constant_one():
    from run_experiment import clean_trajectory_flips

    deltas = [0.0, 0.1, 0.25, 0.5, 1.0]
    const = _Stub(lambda o: 0)
    sensitive = _Stub(lambda o: 1 if o[0] > 0.2 else 0)  # reads poisoned link 0
    for seed in range(100, 105):
        assert all(v == 0 for v in clean_trajectory_flips(const, deltas, seed).values())
    tot = {d: 0 for d in deltas}
    for seed in range(100, 105):
        f = clean_trajectory_flips(sensitive, deltas, seed)
        for d in deltas:
            tot[d] += f[d]
    assert tot[0.0] == 0 and tot[1.0] > 0
    # larger delta never reduces flips for a monotone threshold policy
    assert tot[0.1] <= tot[0.25] <= tot[0.5] <= tot[1.0]
