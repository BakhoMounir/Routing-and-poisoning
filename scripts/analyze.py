"""Summaries, paired effects and figures from results/episodes.csv.

Statistical unit = TRAINING SEED (independent PPO models). Scenario-level
differences are first averaged within a model (paired on identical scenarios),
then summarised across models with a t-interval (n = number of models). Scenario
episodes of one model are not treated as independent replicates of the model.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
METRICS = ["max_utilization", "mean_utilization", "squared_utilization",
           "episode_reward", "overloaded_links", "total_hops"]
C = {"ppo": "#0072B2", "sp": "#555555", "greedy": "#D55E00"}


def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return m, sd, m - h, m + h


def per_model_means(df, cols):
    return df.groupby(["train_seed", "delta"])[cols].mean()


def main():
    d = pd.read_csv(RES / "episodes.csv")
    ppo, sp, gr = (d[d.method == k] for k in ("ppo", "shortest_path", "greedy_telemetry"))
    deltas = sorted(ppo.delta.unique())

    # ---- 1. summary by condition (PPO: mean over models of per-model means)
    rows = []
    for dl in deltas:
        pm = per_model_means(ppo, METRICS).xs(dl, level="delta")
        for m in METRICS:
            mean, sd, lo, hi = tci(pm[m])
            rows.append(dict(method="ppo", delta=dl, metric=m, mean=mean, sd_across_models=sd,
                             ci95_lo=lo, ci95_hi=hi, n_models=len(pm), n_scenarios=ppo.scenario_seed.nunique()))
    for name, sub in (("shortest_path", sp), ("greedy_telemetry", gr)):
        for dl in sorted(sub.delta.unique()):
            s = sub[sub.delta == dl]
            for m in METRICS:
                rows.append(dict(method=name, delta=dl, metric=m, mean=s[m].mean(), sd_across_models=np.nan,
                                 ci95_lo=np.nan, ci95_hi=np.nan, n_models=0, n_scenarios=len(s)))
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "summary_by_condition.csv", index=False)

    # ---- 2. paired effect of poisoning on PPO: (delta - clean), same model & scenario
    key = ["train_seed", "scenario_seed"]
    clean = ppo[ppo.delta == 0.0].set_index(key)
    eff = []
    for dl in deltas[1:]:
        att = ppo[ppo.delta == dl].set_index(key)
        diff = (att[METRICS] - clean[METRICS])
        per_model = diff.groupby(level="train_seed").mean()
        affected = ((att["actions"] != clean["actions"])).groupby(level="train_seed").mean()
        for m in METRICS:
            mean, sd, lo, hi = tci(per_model[m])
            eff.append(dict(delta=dl, metric=m, mean_paired_diff=mean, sd_across_models=sd, ci95_lo=lo,
                            ci95_hi=hi, models_increase=int((per_model[m] > 1e-12).sum()),
                            models_decrease=int((per_model[m] < -1e-12).sum()),
                            models_unchanged=int((per_model[m].abs() <= 1e-12).sum()),
                            n_models=len(per_model)))
        mean, sd, lo, hi = tci(affected)
        eff.append(dict(delta=dl, metric="frac_episodes_with_any_action_change", mean_paired_diff=mean,
                        sd_across_models=sd, ci95_lo=lo, ci95_hi=hi, models_increase=np.nan,
                        models_decrease=np.nan, models_unchanged=np.nan, n_models=len(per_model)))
    eff = pd.DataFrame(eff)
    eff.to_csv(RES / "paired_effects_poisoned_minus_clean_ppo.csv", index=False)

    # ---- 3. PPO(clean) vs shortest path, paired on scenario
    sp0 = sp[sp.delta == 0.0].set_index("scenario_seed")
    cmp_rows = []
    for dl in (0.0, 1.0):
        att = ppo[ppo.delta == dl].set_index(key)
        sp_aligned = sp0.loc[att.index.get_level_values("scenario_seed"), METRICS].set_index(att.index)
        per_model = (att[METRICS] - sp_aligned).groupby(level="train_seed").mean()
        for m in METRICS:
            mean, sd, lo, hi = tci(per_model[m])
            cmp_rows.append(dict(ppo_condition=f"delta={dl}", metric=m, mean_ppo_minus_sp=mean, sd_across_models=sd,
                                 ci95_lo=lo, ci95_hi=hi, models_ppo_higher=int((per_model[m] > 0).sum()),
                                 n_models=len(per_model)))
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(RES / "ppo_vs_shortest_path.csv", index=False)

    # ---- 4. legacy-scenario subset (original 100-119) for continuity with the original CSVs
    leg = d[d.legacy_scenario]
    lt = leg.groupby(["method", "delta"])[METRICS].mean().reset_index()
    lt.to_csv(RES / "summary_legacy_scenarios_100_119.csv", index=False)

    # ---- 5. action-change summary
    ac = ppo[ppo.delta > 0].groupby(["train_seed", "delta"]).agg(
        state_flips=("state_flips", "sum"), traj=("traj_action_changes", "sum"),
        n=("n_decisions", "sum"), pert=("pert_mean_abs", "mean"), active=("pert_active_frac", "mean")).reset_index()
    ac["state_flip_rate"] = ac.state_flips / ac.n
    ac["traj_change_rate"] = ac.traj / ac.n
    ac.to_csv(RES / "action_change_by_model.csv", index=False)

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

    # Fig 1: performance vs delta
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, title in zip(axes, ["max_utilization", "mean_utilization"],
                            ["Mean maximum link utilization (lower = less congestion)",
                             "Mean link utilization (lower = shorter paths)"]):
        pm = per_model_means(ppo, [m])[m].unstack("delta")
        for ts in pm.index:
            ax.plot(pm.columns, pm.loc[ts], color=C["ppo"], alpha=0.15, lw=1)
        stat = np.array([tci(pm[dl]) for dl in pm.columns])
        ax.plot(pm.columns, stat[:, 0], "o-", color=C["ppo"], lw=2, label="PPO (mean of 10 models, 95% CI)")
        ax.fill_between(pm.columns, stat[:, 2], stat[:, 3], color=C["ppo"], alpha=0.2)
        ax.axhline(sp[m].mean(), color=C["sp"], ls="--", lw=1.8, label="Shortest path (ignores telemetry)")
        g = gr.groupby("delta")[m].mean()
        ax.plot(g.index, g.values, "s-.", color=C["greedy"], lw=1.8, label="Greedy-on-telemetry (diagnostic)")
        ax.set_xlabel("Configured poisoning strength δ")
        ax.set_ylabel(m.replace("_", " "))
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_performance_vs_poisoning.png")
    plt.close(fig)

    # Fig 2: action change vs delta
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(axes, ["state_flip_rate", "traj_change_rate"],
                              ["State-level flip rate\n(same clean states, clean vs poisoned obs)",
                               "Trajectory-level change rate\n(rollout under attack vs clean rollout)"]):
        pv = ac.pivot(index="train_seed", columns="delta", values=col) * 100
        for ts in pv.index:
            ax.plot(pv.columns, pv.loc[ts], color=C["ppo"], alpha=0.2, lw=1, marker="o", ms=3)
        stat = np.array([tci(pv[dl]) for dl in pv.columns])
        ax.plot(pv.columns, stat[:, 0], "o-", color=C["ppo"], lw=2, label="mean of 10 models, 95% CI")
        ax.fill_between(pv.columns, stat[:, 2], stat[:, 3], color=C["ppo"], alpha=0.2)
        ax.set_xlabel("Configured poisoning strength δ")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("% of routing decisions that change")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_action_change_vs_poisoning.png")
    plt.close(fig)

    # Fig 3: actual vs configured perturbation
    fig, ax = plt.subplots(figsize=(5, 4))
    pv = ac.groupby("delta")[["pert", "active"]].mean()
    ax.plot([0, 1], [0, 1], color="#999", ls=":", label="actual = configured")
    ax.plot(pv.index, pv.pert, "o-", color=C["ppo"], label="measured mean |obs_poisoned − obs_clean|")
    ax.set_xlabel("Configured poisoning strength δ")
    ax.set_ylabel("Mean absolute change of poisoned entries")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_actual_vs_configured_perturbation.png")
    plt.close(fig)

    # Fig 4: per-model heterogeneity of the effect
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, lab in zip(axes, ["max_utilization", "mean_utilization"],
                          ["Δ mean max utilization (poisoned − clean)", "Δ mean link utilization (poisoned − clean)"]):
        for j, dl in enumerate(deltas[1:]):
            att = ppo[ppo.delta == dl].set_index(key)
            pmv = (att[m] - clean[m]).groupby(level="train_seed").mean()
            ax.scatter(np.full(len(pmv), j) + np.linspace(-0.12, 0.12, len(pmv)), pmv.values, color=C["ppo"], s=18, alpha=0.8)
            mean, _, lo, hi = tci(pmv)
            ax.errorbar(j + 0.28, mean, yerr=[[mean - lo], [hi - mean]], fmt="D", color="k", ms=5, capsize=3)
        ax.axhline(0, color="#999", lw=1)
        ax.set_xticks(range(len(deltas) - 1))
        ax.set_xticklabels([f"δ={x:g}" for x in deltas[1:]])
        ax.set_ylabel(lab)
    axes[0].set_title("Dots = individual trained models (n=10); diamond = mean, 95% CI", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_per_model_effect_heterogeneity.png")
    plt.close(fig)

    # ---- markdown tables for the report
    def md(df, cols, fmt=4):
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(f"{r[c]:.{fmt}f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
        return "\n".join(out)

    parts = ["# Auto-generated summary tables (scripts/analyze.py)\n"]
    t = summ[summ.metric.isin(["max_utilization", "mean_utilization", "episode_reward", "overloaded_links", "total_hops"])]
    parts.append("## Mean by method and delta (PPO: mean of per-model means; 95% t-CI across 10 models)\n")
    parts.append(md(t, ["method", "delta", "metric", "mean", "ci95_lo", "ci95_hi"]))
    parts.append("\n## Paired effect of poisoning on PPO (poisoned − clean)\n")
    parts.append(md(eff, ["delta", "metric", "mean_paired_diff", "ci95_lo", "ci95_hi", "models_increase", "models_decrease", "models_unchanged"]))
    parts.append("\n## PPO − shortest path (paired by scenario)\n")
    parts.append(md(cmp_df, ["ppo_condition", "metric", "mean_ppo_minus_sp", "ci95_lo", "ci95_hi", "models_ppo_higher"]))
    parts.append("\n## Action changes\n")
    a2 = ac.groupby("delta")[["state_flip_rate", "traj_change_rate", "pert", "active"]].agg(["mean", "min", "max"])
    a2.columns = ["_".join(c) for c in a2.columns]
    parts.append(md(a2.reset_index(), list(a2.reset_index().columns)))
    (RES / "summary_tables.md").write_text("\n".join(parts) + "\n")
    print("analysis complete")


if __name__ == "__main__":
    main()
