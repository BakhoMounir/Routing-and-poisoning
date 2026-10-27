import numpy as np
import pytest
from p2.baselines import BASELINES
from p2.envs import EnhanceEnv, ReplaceEnv, Telemetry
from p2.sim import Episode, Network, make_scenario
from p2x.policies import UtilWeightDijkstra, build_baselines, shortest_feasible_k, unit_weight_dijkstra
from p2x.rollout import InvariantViolation, run_baseline, run_model
from p2x.telemetry import ATTACK_KINDS, AdditiveTelemetry, attack_id, episode_seed, make_telemetry
from x_helpers import StubModel, stub_hash


@pytest.fixture(scope="module")
def net(): return Network()


class DemandOnly:
    """Episode proxy exposing ONLY demand(): any access to true state raises, proving policies read just the observation."""
    def __init__(self, ep): self._ep = ep
    def demand(self): return self._ep.demand()
    def __getattr__(self, k): raise AssertionError(f"policy touched true state: ep.{k}")


# ------------------------------------------------------------------ policies
def all_policies():
    return {**build_baselines(dict(p=1.0)), "w_pen": UtilWeightDijkstra(2.0, 0.1, 10.0)}


def test_policies_read_only_observation_and_demand(net):
    sc = make_scenario(3, 2.0); ep = Episode(net, sc)
    for name, fn in all_policies().items():
        e = fn(net, DemandOnly(ep), ep.true_util()); assert len(e) >= 1, name


def test_sf_k_returns_candidate_and_first_feasible(net):
    ep = Episode(net, make_scenario(4, 1.0)); s, d, vol = ep.demand(); cands = net.candidates(s, d)
    e = shortest_feasible_k(net, ep, np.zeros(net.E)); assert any(np.array_equal(e, c) for _, c in cands) and np.array_equal(e, cands[0][1])
    u = np.zeros(net.E); u[cands[0][1]] = 1.0                                   # candidate 0 observed full -> must move on
    e2 = shortest_feasible_k(net, ep, u); assert not np.array_equal(e2, cands[0][1])


def test_unit_weight_is_min_hop_and_ignores_telemetry(net):
    import networkx as nx
    ep = Episode(net, make_scenario(5, 2.0)); s, d, _ = ep.demand()
    a, b = unit_weight_dijkstra(net, ep, np.zeros(net.E)), unit_weight_dijkstra(net, ep, np.ones(net.E))
    assert np.array_equal(a, b) and len(a) == nx.shortest_path_length(net.G, s, d)


def test_heuristic_pipeline_equals_enhance_env_and_p0_is_unit(net):
    ep = Episode(net, make_scenario(6, 2.0)); s, d, vol = ep.demand(); u = np.random.default_rng(0).uniform(0, 1, net.E)
    h = UtilWeightDijkstra(2.0, 0.1, 10.0); act = h.action(net, u, vol)
    assert act.shape == (net.E,) and act.min() >= -1 - 1e-12 and act.max() <= 1 + 1e-12
    env = EnhanceEnv(net=net); env.reset(options={"scenario": make_scenario(6, 2.0)})
    assert np.array_equal(h(net, ep, u), env._edges_for(act))                  # identical to what Agent B's pipeline would do with that action
    assert np.array_equal(UtilWeightDijkstra(0.0)(net, ep, u), unit_weight_dijkstra(net, ep, u))


def test_heuristic_uses_observed_not_true_utilization(net):
    ep = Episode(net, make_scenario(7, 1.0)); s, d, _ = ep.demand(); c0 = net.candidates(s, d)[0][1]
    u = np.zeros(net.E); u[c0] = 0.95                                          # observation says the min-hop path is congested; true state is empty
    assert not np.array_equal(UtilWeightDijkstra(2.0)(net, ep, u), c0)


@pytest.mark.parametrize("bad", [dict(p=-1), dict(p=1, floor=0), dict(p=1, floor=2), dict(p=1, infeasible_multiplier=0.5)])
def test_heuristic_rejects_bad_parameters(bad):
    with pytest.raises(ValueError): UtilWeightDijkstra(**bad)


# ------------------------------------------------------------------ telemetry
@pytest.mark.parametrize("kind", ATTACK_KINDS)
def test_perturbation_budget_range_no_mutation_no_alias(kind):
    rng = np.random.default_rng(1); eps = 0.2
    for _ in range(20):
        true = rng.uniform(0, 1, 42); keep = true.copy(); t = AdditiveTelemetry(kind, eps, 8, seed=3); o = t.observe(true)
        assert np.array_equal(true, keep) and o is not true and not np.shares_memory(o, true)
        assert o.min() >= 0 and o.max() <= 1 and np.abs(o - true).max() <= eps + 1e-12


def test_eps_zero_is_identity_and_clean_matches_base_telemetry():
    u = np.random.default_rng(2).uniform(0, 1, 42)
    for kind in ATTACK_KINDS: assert np.array_equal(AdditiveTelemetry(kind, 0.0).observe(u), u)
    assert np.array_equal(AdditiveTelemetry("clean", 0.3).observe(u), Telemetry().observe(u))


def test_topk_attacks_touch_exactly_the_right_edges():
    u = np.linspace(0.05, 0.95, 42); k = 8
    d = AdditiveTelemetry("conceal_topk", 0.04, k).observe(u) - u; assert set(np.nonzero(d)[0]) == set(np.argsort(-u)[:k]) and np.allclose(d[np.nonzero(d)], -0.04)
    d = AdditiveTelemetry("fake_topk", 0.04, k).observe(u) - u; assert set(np.nonzero(d)[0]) == set(np.argsort(u)[:k]) and np.allclose(d[np.nonzero(d)], 0.04)


def test_seeding_deterministic_and_distinct():
    a, b = make_telemetry("random_sign", 0.2, 8, 123, 2.0), make_telemetry("random_sign", 0.2, 8, 123, 2.0); u = np.full(42, 0.5)
    assert np.array_equal(a.observe(u), b.observe(u)); assert episode_seed("x", 1, 1.0) != episode_seed("x", 2, 1.0) != episode_seed("y", 2, 1.0)
    assert attack_id("clean", 0) == "clean" and attack_id("conceal_topk", 0.2, 8) == "conceal_topk@0.2_k8"


@pytest.mark.parametrize("kind", ATTACK_KINDS)
def test_same_actions_same_true_state_and_reward(net, kind):
    """Poisoned and clean environments driven with identical actions end in IDENTICAL loads and rewards."""
    sc = make_scenario(12, 2.0); c, p = ReplaceEnv(net=net), ReplaceEnv(net=net, telemetry=AdditiveTelemetry(kind, 0.4, 8, 1))
    c.reset(options={"scenario": sc}); p.reset(options={"scenario": sc}); cap0 = net.cap.copy(); rs = []
    for t in range(len(sc.vol)):
        _, rc, *_ = c.step(t % 4); _, rp, *_ = p.step(t % 4); rs.append((rc, rp)); assert np.array_equal(c.ep.loads, p.ep.loads)
    assert all(a == b for a, b in rs), 'rewards differ'
    assert np.array_equal(net.cap, cap0), 'capacities changed'
    assert np.array_equal(c.ep.loads, p.ep.loads), 'final true loads differ'


def test_obs_independent_policy_is_unaffected_by_any_attack(net):
    sc = make_scenario(13, 2.0); base = run_baseline(net, BASELINES["minhop"], sc)
    for kind in ATTACK_KINDS:
        m = run_baseline(net, BASELINES["minhop"], sc, telemetry=make_telemetry(kind, 0.4, 8, sc.seed, 2.0), counterfactual=True)
        assert m["accepted_offered_ratio"] == base["accepted_offered_ratio"] and m["frac_decisions_flipped"] == 0 and m["replay_ok"]


def test_obs_dependent_policies_can_change_but_replay_verifies(net):
    sc = make_scenario(14, 3.0); ch = 0
    for agent in ("replace", "enhance"):
        m = run_model(net, agent, StubModel(agent), sc, telemetry=make_telemetry("conceal_topk", 0.4, 8, sc.seed, 3.0), counterfactual=True)
        assert m["replay_ok"] and m["caps_ok"] and m["tele_max_abs_delta"] <= 0.4 + 1e-9 and 0 <= m["accepted_offered_ratio"] <= 1; ch += m["frac_decisions_flipped"] > 0
    assert ch >= 1, "stub policies should react to perturbed observations in at least one agent"


def test_model_parameters_unchanged_by_poisoned_rollout(net):
    m = StubModel("replace"); h = stub_hash(m); sc = make_scenario(15, 2.0)
    run_model(net, "replace", m, sc, telemetry=make_telemetry("fake_topk", 0.4, 8, sc.seed, 2.0)); assert stub_hash(m) == h


def test_replay_check_detects_a_telemetry_that_corrupts_true_state(net):
    class Evil(Telemetry):
        ep = None
        def observe(self, u):
            if Evil.ep is not None and Evil.ep.t == 5: Evil.ep.loads[:] = 0.0      # tampers with physical state
            return np.array(u, copy=True)
    sc = make_scenario(16, 2.0); ev = Evil()
    def fn(net_, ep, obs):
        Evil.ep = ep; return net.candidates(*ep.demand()[:2])[0][1]
    with pytest.raises(InvariantViolation): run_baseline(net, fn, sc, telemetry=ev)
    Evil.ep = None
