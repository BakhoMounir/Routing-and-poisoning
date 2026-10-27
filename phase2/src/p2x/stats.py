"""Paired-difference statistics that keep the two sources of uncertainty separate.

D has shape (n_models, n_scenarios): D[i, j] = metric(test; model i, scenario j) - metric(reference; scenario j [, model i]).
Non-learned methods have n_models == 1. All outputs are in PERCENTAGE POINTS (x100) of the accepted/offered ratio.
 * ci_scen_*  : scenario-sampling uncertainty, CONDITIONAL on the trained models actually used (bootstrap over scenarios).
 * ci_model_t : training-seed variability of the per-model mean difference (t-interval, needs >= 2 models).
 * ci_cross_* : both together (two-way bootstrap: models AND scenarios resampled independently); slightly conservative.
Scenarios at different loads share seeds, so each load is analysed separately; Holm is valid under arbitrary dependence.
"""
from __future__ import annotations

import numpy as np


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _counts(rng, n, B):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B) / n          # (B, n) resampling weights


def cross_boot_means(D, B, rng, chunk=2000):
    D = np.asarray(D, dtype=float)
    m, s = D.shape
    out = []
    for lo in range(0, B, chunk):
        b = min(chunk, B - lo)
        wm, ws = _counts(rng, m, b), _counts(rng, s, b)
        out.append(((wm @ D) * ws).sum(axis=1))
    return np.concatenate(out)


def scen_boot_means(D, B, rng):
    return _counts(rng, D.shape[1], B) @ np.asarray(D, float).mean(axis=0)


def _t_crit(df, level):
    try:
        from scipy import stats as st
        return float(st.t.ppf(0.5 + level / 2, df))
    except ImportError:                                                   # pragma: no cover
        return 1.96


def summarize_diff(D, B=10000, alpha=0.05, n_tests=1, rng=None) -> dict:
    """Point estimate, three kinds of CI, bootstrap p-value (two-sided) and per-model/scenario spreads, in pp."""
    rng = rng or np.random.default_rng(0)
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.size == 0 or not np.all(np.isfinite(D)):
        raise ValueError("D must be a non-empty finite 2-D array (models x scenarios)")
    m, s = D.shape
    pp = 100.0
    mean = D.mean()
    cross = cross_boot_means(D, B, rng)
    scen = scen_boot_means(D, B, rng)
    q = lambda x, a: (float(np.percentile(x, 100 * a / 2)) * pp, float(np.percentile(x, 100 * (1 - a / 2))) * pp)
    c_lo, c_hi = q(cross, alpha)
    cb_lo, cb_hi = q(cross, alpha / max(n_tests, 1))
    s_lo, s_hi = q(scen, alpha)
    pm = D.mean(axis=1)
    if m > 1:
        h = _t_crit(m - 1, 1 - alpha) * pm.std(ddof=1) / np.sqrt(m)
        t_lo, t_hi, sd_m = (pm.mean() - h) * pp, (pm.mean() + h) * pp, float(pm.std(ddof=1)) * pp
    else:
        t_lo = t_hi = sd_m = float("nan")
    c_le, c_ge = int((cross <= 0).sum()), int((cross >= 0).sum())
    p = min(1.0, 2.0 * (min(c_le, c_ge) + 1) / (B + 1))
    return dict(mean_pp=float(mean) * pp, ci_cross_lo=c_lo, ci_cross_hi=c_hi, ci_cross_bonf_lo=cb_lo, ci_cross_bonf_hi=cb_hi,
                ci_scen_lo=s_lo, ci_scen_hi=s_hi, ci_model_t_lo=t_lo, ci_model_t_hi=t_hi,
                sd_across_models_pp=sd_m, sd_across_scenarios_pp=float(D.mean(axis=0).std(ddof=1)) * pp if s > 1 else float("nan"),
                p_boot=p, n_models=m, n_scenarios=s, n_tests_in_family=n_tests)
