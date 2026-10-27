#!/usr/bin/env python3
"""Generate reproducible diagnostic figures from existing Phase 2 results.

These plots are descriptive diagnostics, not confirmatory evaluation results.
No model training or evaluation is performed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT = Path(__file__).resolve()
PHASE2 = SCRIPT.parents[1]
RESULTS = PHASE2 / "results" / "v2"
DEFAULT_OUT = PHASE2 / "figures"


def save_figure(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")


def plot_training_curves(out_dir):
    """Plot validation history for available replace-agent training seeds."""
    root = RESULTS / "models" / "replace_600k"
    curves = []

    for path in sorted(root.glob("seed*/val_curve.csv")):
        df = pd.read_csv(path)
        required = {"step", "val"}
        if not required.issubset(df.columns):
            print(f"SKIP malformed validation CSV: {path}")
            continue

        df = df[["step", "val"]].apply(pd.to_numeric, errors="coerce").dropna()
        if not df.empty:
            df["seed"] = path.parent.name.removeprefix("seed")
            curves.append(df)

    if not curves:
        print("SKIP training curves: no usable validation CSV files")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for df in curves:
        ax.plot(df["step"], df["val"], alpha=0.30, linewidth=1)

    # Aggregate by training step only where multiple seeds have observations.
    all_steps = sorted(set().union(*(set(df["step"]) for df in curves)))
    matrix = pd.DataFrame(index=all_steps)

    for df in curves:
        # Interpolate only within each seed's observed training range.
        steps = df["step"].to_numpy(dtype=float)
        vals = df["val"].to_numpy(dtype=float)
        matrix[df["seed"].iloc[0]] = np.interp(
            np.asarray(all_steps, dtype=float), steps, vals,
            left=np.nan, right=np.nan
        )

    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1, ddof=1)
    ax.plot(mean.index, mean, linewidth=2.5, label="Across-seed mean")
    ax.fill_between(
        mean.index.to_numpy(dtype=float),
        (mean - std).to_numpy(dtype=float),
        (mean + std).to_numpy(dtype=float),
        alpha=0.18, label="±1 SD across seeds"
    )
    ax.set(
        title="Agent A training: validation score",
        xlabel="Training steps",
        ylabel="Validation score",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, "fig1_agent_a_training_curves")


def plot_baselines(out_dir):
    """Plot diagnostic baseline results by offered traffic load."""
    path = RESULTS / "diag_baselines" / "episodes.csv"
    if not path.exists():
        print(f"SKIP baseline plot: missing {path}")
        return

    df = pd.read_csv(path)
    required = {"method", "load", "accepted_offered_ratio"}
    if not required.issubset(df.columns):
        print("SKIP baseline plot: required columns are missing")
        return

    df = df.copy()
    for col in ("load", "accepted_offered_ratio"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["load", "accepted_offered_ratio"])

    if "replay_ok" in df.columns:
        valid = df["replay_ok"].astype(str).str.lower().eq("true")
        if not valid.all():
            print("WARNING: invalid replay rows excluded from baseline plot")
            df = df.loc[valid]

    if df.empty:
        print("SKIP baseline plot: no valid rows")
        return

    # Episode-level intervals are descriptive and do not account for
    # model-seed uncertainty. These data are diagnostic, not confirmatory.
    stats = (
        df.groupby(["method", "load"])["accepted_offered_ratio"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    stats["se"] = stats["std"].fillna(0) / np.sqrt(stats["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, group in stats.groupby("method"):
        group = group.sort_values("load")
        ax.errorbar(
            group["load"], group["mean"],
            yerr=1.96 * group["se"],
            marker="o", capsize=3, linewidth=1.8, label=str(method)
        )

    ax.set(
        title="Routing baseline diagnostics by traffic load",
        xlabel="Traffic load",
        ylabel="Accepted/offered traffic-volume ratio",
        ylim=(0, 1.05),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig2_baseline_diagnostics")


def plot_checkpoint_comparison(out_dir):
    """Compare best-checkpoint diagnostic scores across model seeds."""
    paths = {
        "Agent A (replace)": RESULTS / "diag_checkpoints_replace" / "per_model.csv",
        "Agent B (enhance)": RESULTS / "diag_checkpoints_enhance" / "per_model.csv",
    }
    frames = []

    for label, path in paths.items():
        if not path.exists():
            print(f"SKIP missing checkpoint results: {path}")
            continue

        df = pd.read_csv(path)
        if not {"seed", "ckpt", "mean_ratio"}.issubset(df.columns):
            print(f"SKIP malformed checkpoint results: {path}")
            continue

        df = df[df["ckpt"].astype(str).str.lower().eq("best")].copy()
        df["mean_ratio"] = pd.to_numeric(df["mean_ratio"], errors="coerce")
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        df = df.dropna(subset=["mean_ratio", "seed"])
        df["agent_label"] = label
        frames.append(df)

    if not frames:
        print("SKIP checkpoint comparison: no usable per-model files")
        return

    df = pd.concat(frames, ignore_index=True)
    labels = list(df["agent_label"].unique())
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, label in enumerate(labels):
        group = df[df["agent_label"] == label].sort_values("seed")
        x = np.full(len(group), idx, dtype=float)
        # Show individual seeds; avoid implying these are scenario-level CIs.
        ax.scatter(x, group["mean_ratio"], s=42, alpha=0.8, label=None)
        mean = group["mean_ratio"].mean()
        sd = group["mean_ratio"].std(ddof=1) if len(group) > 1 else 0.0
        ax.errorbar(
            idx, mean, yerr=sd, fmt="D", color="black",
            capsize=5, markersize=6, label="Mean ± SD across seeds"
            if idx == 0 else None
        )
        for _, row in group.iterrows():
            ax.annotate(
                str(int(row["seed"])), (idx, row["mean_ratio"]),
                xytext=(4, 3), textcoords="offset points", fontsize=7
            )

    ax.set_xticks(range(len(labels)), labels)
    ax.set(
        title="Best-checkpoint diagnostic scores across seeds",
        ylabel="Accepted/offered traffic-volume ratio",
        xlabel="Agent",
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, "fig3_checkpoint_diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT,
        help="Output directory for PNG and PDF figures",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()

    # Matplotlib defaults are intentionally used for reproducibility
    # across environments; no seaborn dependency is required.
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    plot_training_curves(out_dir)
    plot_baselines(out_dir)
    plot_checkpoint_comparison(out_dir)
    print(f"\nFigure output directory: {out_dir}")


if __name__ == "__main__":
    main()
