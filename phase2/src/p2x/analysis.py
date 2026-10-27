"""Tables for the confirmatory and poisoning stages. pandas is imported lazily. All effects in percentage points of accepted_offered_ratio."""
from __future__ import annotations

import numpy as np

from .stats import holm, summarize_diff

M = "accepted_offered_ratio"


def matrix(df, method, load, exclude_seeds=(), extra=None):
    """(models x scenarios) matrix for one method/load. Non-learned methods have one row (model_seed == -1)."""
    q = df[(df.method == method) & (df.load == load)]
    if extra:
        for k, v in extra.items():
            q = q[q[k] == v]
    if len(exclude_seeds):
        q = q[~q.model_seed.isin(list(exclude_seeds))]
    if q.empty:
        return None
    pv = q.pivot_table(index="model_seed", columns="scenario_idx", values=M, aggfunc="first").sort_index(axis=1)
    if pv.isna().any().any():
        raise ValueError(f"incomplete grid for {method} load {load}: some (model, scenario) cells missing")
    return pv.to_numpy()


def paired_D(T, R):
    """Difference matrix test - reference. Reference with several models enters as its per-scenario model-average (independent models are not paired)."""
    if T.shape[1] != R.shape[1]:
        raise ValueError("test and reference cover different scenario sets")
    return T - R.mean(axis=0, keepdims=True)


def compare(df, pairs, loads, alpha, B, rng, exclude_seeds=(), family="primary"):
    """pairs: [(id, test, ref)] evaluated per load. Holm across all computable tests in the family; Bonferroni-adjusted CI at alpha/n."""
    import pandas as pd
    cells = []
    for pid, t, r in pairs:
        for L in loads:
            T, R = matrix(df, t, L, exclude_seeds), matrix(df, r, L, exclude_seeds)
            if T is None or R is None:
                cells.append(dict(id=pid, test=t, ref=r, load=L, computed=False))
            else:
                cells.append(dict(id=pid, test=t, ref=r, load=L, computed=True, D=paired_D(T, R)))
    n = sum(c["computed"] for c in cells)
    out = []
    for c in cells:
        if not c["computed"]:
            out.append(dict(id=c["id"], test=c["test"], ref=c["ref"], load=c["load"], family=family, computed=False, note="method absent from episodes"))
            continue
        s = summarize_diff(c["D"], B=B, alpha=alpha, n_tests=n, rng=rng)
        out.append(dict(id=c["id"], test=c["test"], ref=c["ref"], load=c["load"], family=family, computed=True, **s))
    df_out = pd.DataFrame(out)
    if n:
        m = df_out.computed.to_numpy(bool)
        df_out["p_holm"] = np.nan
        df_out.loc[m, "p_holm"] = holm(df_out.loc[m, "p_boot"].to_numpy())
        df_out["reject_holm_alpha"] = df_out.p_holm < alpha if family == "primary" else np.nan
    return df_out


def summary_table(df, loads, exclude_seeds=()):
    import pandas as pd
    rows = []
    for m in sorted(df.method.unique()):
        for L in loads:
            X = matrix(df, m, L, exclude_seeds)
            if X is None:
                continue
            pm = X.mean(axis=1)
            rows.append(dict(method=m, load=L, mean=float(X.mean()), n_models=X.shape[0], n_scenarios=X.shape[1],
                             sd_across_models=float(pm.std(ddof=1)) if len(pm) > 1 else np.nan,
                             sd_across_scenarios=float(X.mean(axis=0).std(ddof=1))))
    return pd.DataFrame(rows)


def poison_tables(df, targets, attacks, eps_list, loads, primary, alpha, B, rng, exclude_seeds=()):
    """Delta = attacked - clean for the SAME model and scenario (paired); contrast = attacked - random_sign at equal eps."""
    import pandas as pd
    def mat(t, L, aid):
        return matrix(df, t, L, exclude_seeds, extra={"attack": aid})
    eff, prim_cells = [], []
    for t in targets:
        for L in loads:
            C = mat(t, L, "clean")
            if C is None:
                continue
            for a in attacks:
                for e in eps_list:
                    A = mat(t, L, f"{a}@{e:g}")
                    if A is None:
                        continue
                    row = dict(target=t, load=L, attack=a, eps=e, **summarize_diff(A - C, B=B, alpha=alpha, n_tests=1, rng=rng))
                    Rn = mat(t, L, f"random_sign@{e:g}")
                    if Rn is not None and a != "random_sign":
                        c = summarize_diff(A - Rn, B=B, alpha=alpha, n_tests=1, rng=rng)
                        row.update(contrast_mean_pp=c["mean_pp"], contrast_ci_lo=c["ci_cross_lo"], contrast_ci_hi=c["ci_cross_hi"], contrast_p=c["p_boot"])
                    eff.append(row)
                    if (t in primary["targets"] and a in primary["attacks"] and abs(e - primary["eps"]) < 1e-12 and L == primary["load"] and "contrast_p" in row):
                        prim_cells.append(len(eff) - 1)
    out = pd.DataFrame(eff)
    out["primary"] = False
    if prim_cells and len(out):
        out.loc[prim_cells, "primary"] = True
        out["contrast_p_holm"] = np.nan
        out.loc[prim_cells, "contrast_p_holm"] = holm(out.loc[prim_cells, "contrast_p"].to_numpy())
    return out
