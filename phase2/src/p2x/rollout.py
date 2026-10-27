"""Episode runners with an explicit telemetry channel and a replay invariant.

Every finished episode is re-simulated from the recorded routes on a FRESH Episode with no telemetry; the true
loads and metrics must match exactly. Together with unchanged capacities this proves that a perturbed
observation influenced the outcome only through the routes the policy chose, never through simulator state or reward.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from p2.envs import EnhanceEnv, ReplaceEnv, Telemetry, build_features
from p2.sim import Episode

ENVS = {"replace": ReplaceEnv, "enhance": EnhanceEnv}
RENAME = {"delivery_ratio": "accepted_offered_ratio", "blocking_ratio": "blocked_volume_ratio",
          "delivered": "accepted_volume", "offered": "offered_volume"}
METRIC = "accepted_offered_ratio"


class InvariantViolation(AssertionError):
    """True simulator state / capacities / reward were affected by something other than the chosen routes."""


def _finalize(net, scen, ep, routes, cap0, tele, same0, flips, n):
    m = {RENAME.get(k, k): v for k, v in ep.metrics().items()}
    rep = Episode(net, scen)
    for e in routes:
        rep.route(e)
    replay_ok = bool(np.array_equal(rep.loads, ep.loads) and rep.accepted_total == ep.accepted_total)
    caps_ok = bool(np.array_equal(net.cap, cap0))
    if not (replay_ok and caps_ok and ep.total_offered == float(scen.vol.sum())):
        raise InvariantViolation(f"replay_ok={replay_ok} caps_ok={caps_ok} (scenario seed {scen.seed}, load {scen.load})")
    m.update(replay_ok=replay_ok, caps_ok=caps_ok, frac_route_is_minhop_cand0=same0 / n)
    if flips is not None:
        m["frac_decisions_flipped"] = flips / n
    if tele is not None and hasattr(tele, "stats"):
        m.update(tele.stats())
    return m


def run_baseline(net, fn, scen, telemetry=None, counterfactual=False):
    """fn(net, ep, obs_util) -> edges. obs_util comes ONLY from telemetry.observe(true_util)."""
    tele = telemetry if telemetry is not None else Telemetry()
    cap0 = net.cap.copy()
    ep = Episode(net, scen)
    routes, same0, flips = [], 0, (0 if counterfactual else None)
    while not ep.done:
        s, d, _ = ep.demand()
        true = ep.true_util()
        e = np.asarray(fn(net, ep, tele.observe(true)), dtype=np.int64)
        if counterfactual:
            flips += int(not np.array_equal(e, np.asarray(fn(net, ep, true), dtype=np.int64)))
        same0 += int(np.array_equal(e, net.candidates(s, d)[0][1]))
        routes.append(e)
        ep.route(e)
    return _finalize(net, scen, ep, routes, cap0, tele, same0, flips, len(scen.vol))


def run_model(net, agent, model, scen, telemetry=None, counterfactual=False):
    """model: anything with predict(obs, deterministic=True) -> (action, state), e.g. an SB3 PPO."""
    cap0 = net.cap.copy()
    env = ENVS[agent](net=net, telemetry=telemetry)
    obs, _ = env.reset(options={"scenario": scen})
    routes, same0, flips, done, info = [], 0, (0 if counterfactual else None), False, {}
    with_cand = agent == "replace"
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        s, d, _ = env.ep.demand()
        e = np.asarray(env._edges_for(a), dtype=np.int64)
        if counterfactual:
            a_cf, _ = model.predict(build_features(net, env.ep, env.last_true_util, with_cand), deterministic=True)
            flips += int(not np.array_equal(e, np.asarray(env._edges_for(a_cf), dtype=np.int64)))
        same0 += int(np.array_equal(e, net.candidates(s, d)[0][1]))
        routes.append(e)
        obs, _, done, _, info = env.step(a)
    m = _finalize(net, scen, env.ep, routes, cap0, telemetry, same0, flips, len(scen.vol))
    if agent == "replace":
        m["action_hist"] = str(np.bincount(np.asarray([_cand_index(net, scen, i, r) for i, r in enumerate(routes)]), minlength=4).tolist())
    return m


def _cand_index(net, scen, t, edges):
    s, d = int(scen.src[t]), int(scen.dst[t])
    for k, (_, e) in enumerate(net.candidates(s, d)):
        if np.array_equal(e, edges):
            return k
    return 0


def evaluate_fast(net, agent, model, scens):
    """Lean validation loop (no invariants): list of accepted_offered_ratio per scenario, clean telemetry."""
    env = ENVS[agent](net=net, fixed_scenarios=list(scens))
    out = []
    for _ in scens:
        obs, _ = env.reset()
        done, info = False, {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(a)
        out.append(info["delivery_ratio"])
    return out


def collect_rows(method, model_seed, scen_list, run_one):
    """run_one(scenario) -> metrics dict. scen_list = [(idx, load, Scenario)]."""
    rows = []
    for i, L, sc in scen_list:
        rows.append(dict(method=method, model_seed=model_seed, scenario_idx=i, scenario_seed=sc.seed, load=L, **run_one(sc)))
    return rows


# ---------------------------------------------------------------- checkpoints (SB3 imported lazily)
def load_model(models_dir, seed, ckpt):
    p = Path(models_dir) / f"seed{seed}" / f"{ckpt}.zip"
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    try:
        from stable_baselines3 import PPO
    except ImportError as e:                                   # pragma: no cover
        raise RuntimeError("stable-baselines3/torch are required for checkpoint loading; activate the project venv "
                           "(see requirements.txt). Nothing was installed automatically.") from e
    return PPO.load(str(p), device="cpu")


def param_sha256(model) -> str:
    """Same recipe as scripts/train_ppo.py (final_param_sha256 in meta.json)."""
    import torch
    flat = torch.cat([p.flatten() for p in model.policy.parameters()]).detach().numpy()
    return hashlib.sha256(flat.tobytes()).hexdigest()
