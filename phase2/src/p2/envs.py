"""Gymnasium environments. TRUE state lives in sim.Episode; the agent only receives `observe()`.

Telemetry channel: `Telemetry.observe(true_util)` returns what the agent SEES. It is the identity
(a copy) in the clean stage. It is the single place a future test-time perturbation would act, and
it can never write back to the Episode (tests assert this). No attack is implemented in this stage.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .sim import (K_PATHS, N_FLOWS, TRAIN_SEED_RANGE, VOL_NORM, Episode, Network, make_scenario)


class Telemetry:
    def observe(self, true_util):
        return np.array(true_util, dtype=np.float64, copy=True)


def build_features(net, ep, obs_util, with_candidates):
    """Observation = f(OBSERVED utilization, current demand, progress). No hidden state is used."""
    s, d, vol = ep.demand()
    src = np.zeros(net.n); dst = np.zeros(net.n); src[s] = 1; dst[d] = 1
    parts = [np.clip(obs_util, 0, 1), src, dst, [min(vol / VOL_NORM, 1.0)], [1.0 - ep.t / len(ep.scen.vol)]]
    if with_candidates:
        resid_frac = 1.0 - np.clip(obs_util, 0, 1)
        for _, e in net.candidates(s, d):
            parts.append([float(resid_frac[e].min()), len(e) / net.max_candidate_hops])
    return np.concatenate(parts).astype(np.float32)


class _Base(gym.Env):
    metadata = {"render_modes": []}
    with_candidates = False

    def __init__(self, fixed_scenarios=None, load_range=(1.0, 3.0), telemetry=None, net=None):
        super().__init__()
        self.net = net or Network()
        self.telemetry = telemetry or Telemetry()
        self.fixed = fixed_scenarios          # optional list of Scenario objects, cycled in order
        self.load_range = load_range
        self._i = 0
        size = self.net.E + 2 * self.net.n + 2 + (2 * K_PATHS if self.with_candidates else 0)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)
        self.ep = None; self.last_true_util = None; self.last_obs_util = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            scen = options["scenario"]
        elif self.fixed is not None:
            scen = self.fixed[self._i % len(self.fixed)]; self._i += 1
        else:
            sd = int(self.np_random.integers(*TRAIN_SEED_RANGE))
            load = float(self.np_random.uniform(*self.load_range))
            scen = make_scenario(sd, load)
        self.ep = Episode(self.net, scen)
        return self._obs(), {}

    def _obs(self):
        tu = self.ep.true_util()
        self.last_true_util = tu
        self.last_obs_util = self.telemetry.observe(tu)
        return build_features(self.net, self.ep, self.last_obs_util, self.with_candidates)

    def _edges_for(self, action):
        raise NotImplementedError

    def step(self, action):
        edges = self._edges_for(action)
        before = self.ep.accepted_total
        acc = self.ep.route(edges)
        reward = acc / self.ep.total_offered           # TRUE accepted volume, never telemetry
        done = self.ep.done
        info = self.ep.metrics() if done else {}
        obs = np.zeros(self.observation_space.shape, np.float32) if done else self._obs()
        return obs, float(reward), done, False, info


class ReplaceEnv(_Base):
    """Agent A (DRL replacement): PPO picks one of K candidate paths."""
    with_candidates = True

    def __init__(self, **kw):
        super().__init__(**kw); self.action_space = spaces.Discrete(K_PATHS)

    def _edges_for(self, action):
        a = int(action)
        if not 0 <= a < K_PATHS: raise ValueError(f"invalid action {a}")
        s, d, _ = self.ep.demand()
        return self.net.candidates(s, d)[a][1]


W_LN = float(np.log(10.0))


class EnhanceEnv(_Base):
    """Agent B (DRL enhancement): PPO outputs one weight per directed edge; Dijkstra routes on them."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.net.E,), dtype=np.float32)

    @staticmethod
    def weights(action):
        a = np.asarray(action, dtype=np.float64)
        if a.shape[-1:] != (42,) or not np.all(np.isfinite(a)): raise ValueError("invalid weight action")
        return np.exp(np.clip(a, -1.0, 1.0) * W_LN)

    def _edges_for(self, action):
        s, d, _ = self.ep.demand()
        return self.net.dijkstra(s, d, self.weights(action))
