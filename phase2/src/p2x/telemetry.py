"""Test-time telemetry corruption (threat model in docs/THREAT_MODEL.md).

ONLY the utilization vector OBSERVED by a policy is perturbed:   obs = clip(true + delta, 0, 1),  |delta_e| <= eps.
`observe` never mutates its input, never returns an alias of it, and holds no reference to the Episode, the
capacities, the reward or the model, so it cannot change physical state by construction (tests enforce this).
"""
from __future__ import annotations

import hashlib

import numpy as np

from p2.envs import Telemetry

ATTACK_KINDS = ("clean", "random_sign", "bias_all_plus", "bias_all_minus", "conceal_topk", "fake_topk")


def attack_id(kind: str, eps: float, k: int | None = None) -> str:
    if kind == "clean":
        return "clean"
    return f"{kind}@{eps:g}" + (f"_k{k}" if kind.endswith("topk") and k is not None else "")


def episode_seed(attack: str, scenario_seed: int, load: float) -> int:
    """Deterministic per-episode RNG seed, independent of process, order and model."""
    h = hashlib.sha256(f"{attack}|{scenario_seed}|{load:g}".encode()).hexdigest()
    return int(h[:8], 16)


class AdditiveTelemetry(Telemetry):
    """Additive, L-infinity-budgeted corruption of observed utilization.

    random_sign    : delta_e = +/- eps, i.i.d. signs per observation         (noise CONTROL at the same budget)
    bias_all_plus  : delta_e = +eps on every edge                              (uniform over-reporting)
    bias_all_minus : delta_e = -eps on every edge                              (uniform under-reporting)
    conceal_topk   : delta_e = -eps on the k most-utilized edges (TRUE util)   (hide congestion; attacker sees true util)
    fake_topk      : delta_e = +eps on the k least-utilized edges (TRUE util)  (invent congestion; attacker sees true util)
    clean          : identity
    """

    def __init__(self, kind: str, eps: float = 0.0, k: int = 8, seed: int = 0):
        if kind not in ATTACK_KINDS:
            raise ValueError(f"unknown attack kind {kind!r}; choose from {ATTACK_KINDS}")
        if eps < 0 or eps > 1:
            raise ValueError("eps must be in [0, 1]")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.kind, self.eps, self.k = kind, float(eps), int(k)
        self.rng = np.random.default_rng(seed)
        self.n_obs = 0
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.n_changed = 0

    def _delta(self, true: np.ndarray) -> np.ndarray:
        e = self.eps
        if self.kind == "clean" or e == 0.0:
            return np.zeros_like(true)
        if self.kind == "random_sign":
            return e * self.rng.choice(np.array([-1.0, 1.0]), size=true.shape)
        if self.kind == "bias_all_plus":
            return np.full_like(true, e)
        if self.kind == "bias_all_minus":
            return np.full_like(true, -e)
        d = np.zeros_like(true)
        if self.kind == "conceal_topk":
            d[np.argsort(-true, kind="stable")[: self.k]] = -e
        else:  # fake_topk
            d[np.argsort(true, kind="stable")[: self.k]] = +e
        return d

    def observe(self, true_util):
        true = np.array(true_util, dtype=np.float64, copy=True)       # private copy: input is never touched
        obs = np.clip(true + self._delta(true), 0.0, 1.0)             # new array, never an alias of `true_util`
        diff = np.abs(obs - true)
        self.n_obs += 1
        self.max_abs = max(self.max_abs, float(diff.max()))
        self.sum_abs += float(diff.mean())
        self.n_changed += int(diff.max() > 0)
        return obs

    def stats(self) -> dict:
        return dict(tele_n_obs=self.n_obs, tele_max_abs_delta=self.max_abs,
                    tele_mean_abs_delta=self.sum_abs / max(self.n_obs, 1),
                    tele_frac_obs_changed=self.n_changed / max(self.n_obs, 1))


def make_telemetry(kind: str, eps: float, k: int, scenario_seed: int, load: float) -> AdditiveTelemetry:
    aid = attack_id(kind, eps, k)
    return AdditiveTelemetry(kind, eps, k, seed=episode_seed(aid, scenario_seed, load))
