import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from utils.data import MODEL_NAME

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SAE_COLOR    = "#4878d0"
LOGREG_COLOR = "#ee854a"


def load_results(results_dir: str):
    # load selectivity and feature-reuse CSVs from a prior feature_comparison run
    sel_path   = os.path.join(results_dir, "selectivity_results.csv")
    reuse_path = os.path.join(results_dir, "feature_reuse_results.csv")
    if not os.path.exists(sel_path):
        raise FileNotFoundError(
            f"{sel_path} not found.\n"
            "Run interpretability/feature_comparison.py first."
        )
    sel_df   = pd.read_csv(sel_path)
    reuse_df = pd.read_csv(reuse_path) if os.path.exists(reuse_path) else pd.DataFrame()
    return sel_df, reuse_df


def plot_selectivity(sel_df: pd.DataFrame, out_dir: str):
    # violin plots of per-rank Gini distributions and a per-dataset scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Feature Selectivity: SAE Latents vs LogReg Activation Dims",
                 fontsize=13, y=1.01)

    ax = axes[0]
    ranks = sorted(sel_df["rank"].unique())

    positions_sae    = [r * 2.5          for r in ranks]
    positions_logreg = [r * 2.5 + 1.0   for r in ranks]

    vp_sae = ax.violinplot(
        [sel_df.loc[sel_df["rank"] == r, "sae_gini"].dropna().values for r in ranks],
        positions=positions_sae,
        widths=0.8, showmedians=True,
    )
    vp_lr = ax.violinplot(
        [sel_df.loc[sel_df["rank"] == r, "logreg_gini"].dropna().values for r in ranks],
        positions=positions_logreg,
        widths=0.8, showmedians=True,
    )

    for body in vp_sae["bodies"]:
        body.set_facecolor(SAE_COLOR)
        body.set_alpha(0.7)
    for body in vp_lr["bodies"]:
        body.set_facecolor(LOGREG_COLOR)
        body.set_alpha(0.7)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp_sae[part].set_color(SAE_COLOR)
        vp_lr[part].set_color(LOGREG_COLOR)

    tick_positions = [(s + l) / 2 for s, l in zip(positions_sae, positions_logreg)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"rank {r+1}" for r in ranks], rotation=45, ha="right")
    ax.set_ylabel("Gini coefficient")
    ax.set_xlabel("Feature rank (most to least discriminative)")
    ax.set_title("Gini distribution per rank position")

    legend_patches = [
        plt.matplotlib.patches.Patch(color=SAE_COLOR,    label="SAE latents"),
        plt.matplotlib.patches.Patch(color=LOGREG_COLOR, label="LogReg dims"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    ax2 = axes[1]
    per_ds = (
        sel_df.groupby("dataset")[["sae_gini", "logreg_gini"]]
        .mean()
        .dropna()
    )

    ax2.scatter(
        per_ds["logreg_gini"], per_ds["sae_gini"],
        alpha=0.6, s=30, color=SAE_COLOR, edgecolors="none",
    )

    lo = min(per_ds["logreg_gini"].min(), per_ds["sae_gini"].min()) - 0.02
    hi = max(per_ds["logreg_gini"].max(), per_ds["sae_gini"].max()) + 0.02
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1, label="Equal selectivity")

    n_above = (per_ds["sae_gini"] > per_ds["logreg_gini"]).sum()
    n_total = len(per_ds)
    ax2.set_xlabel("LogReg mean Gini (per dataset)")
    ax2.set_ylabel("SAE mean Gini (per dataset)")
    ax2.set_title(
        f"Per-dataset mean Gini\n"
        f"SAE more selective: {n_above}/{n_total} datasets ({n_above/n_total:.0%})"
    )
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "selectivity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")

    fig2, ax3 = plt.subplots(figsize=(6, 4))
    ax3.set_title("Fraction of examples where top feature is active")

    per_ds2 = (
        sel_df.groupby("dataset")[["sae_frac_active", "logreg_frac_active"]]
        .mean()
        .dropna()
    )
    data = [per_ds2["sae_frac_active"].values, per_ds2["logreg_frac_active"].values]
    bp = ax3.boxplot(data, labels=["SAE latents", "LogReg dims"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(SAE_COLOR)
    bp["boxes"][1].set_facecolor(LOGREG_COLOR)
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax3.set_ylabel("Mean fraction active (per dataset)")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    out2 = os.path.join(out_dir, "frac_active.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out2}")


def plot_feature_reuse(reuse_df: pd.DataFrame, out_dir: str):
    # bar chart of per-group Jaccard means and a violin of the full distribution
    if reuse_df.empty:
        print("No feature-reuse data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-task Feature Reuse: Jaccard Similarity of Top-k Features",
                 fontsize=13, y=1.01)

    ax = axes[0]
    group_means = (
        reuse_df.groupby("group")[["sae_jaccard", "logreg_jaccard"]]
        .mean()
        .sort_values("sae_jaccard", ascending=False)
    )

    x    = np.arange(len(group_means))
    w    = 0.35
    ax.bar(x - w / 2, group_means["sae_jaccard"],
           width=w, color=SAE_COLOR,    alpha=0.8, label="SAE latents")
    ax.bar(x + w / 2, group_means["logreg_jaccard"],
           width=w, color=LOGREG_COLOR, alpha=0.8, label="LogReg dims")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace("_", "\n") for g in group_means.index],
        fontsize=8, rotation=45, ha="right",
    )
    ax.set_ylabel("Mean Jaccard similarity")
    ax.set_title("Per-group mean Jaccard (higher = more feature reuse)")
    ax.legend()

    for xi, (sg, lg) in enumerate(
        zip(group_means["sae_jaccard"], group_means["logreg_jaccard"])
    ):
        delta = sg - lg
        color = "green" if delta > 0 else "red"
        ax.text(xi, max(sg, lg) + 0.005, f"{delta:+.2f}",
                ha="center", va="bottom", fontsize=7, color=color)

    ax2 = axes[1]
    data = [reuse_df["sae_jaccard"].values, reuse_df["logreg_jaccard"].values]
    bp = ax2.violinplot(data, positions=[1, 2], showmedians=True, widths=0.6)
    bp["bodies"][0].set_facecolor(SAE_COLOR)
    bp["bodies"][1].set_facecolor(LOGREG_COLOR)
    for body in bp["bodies"]:
        body.set_alpha(0.7)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        bp[part].set_color("black")

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["SAE latents", "LogReg dims"])
    ax2.set_ylabel("Jaccard similarity")
    ax2.set_title(
        f"Pairwise Jaccard distribution\n"
        f"(n={len(reuse_df)} dataset pairs across semantic groups)"
    )

    for pos, vals, color in zip(
        [1, 2],
        [reuse_df["sae_jaccard"], reuse_df["logreg_jaccard"]],
        [SAE_COLOR, LOGREG_COLOR],
    ):
        ax2.text(pos, vals.max() + 0.005, f"μ={vals.mean():.3f}",
                 ha="center", va="bottom", color=color, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "feature_reuse.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")

    fig3, ax3 = plt.subplots(figsize=(5, max(4, len(group_means) * 0.4)))
    deltas = group_means["sae_jaccard"] - group_means["logreg_jaccard"]
    colors = [SAE_COLOR if d > 0 else LOGREG_COLOR for d in deltas]
    ax3.barh(range(len(deltas)), deltas.values, color=colors, alpha=0.8)
    ax3.set_yticks(range(len(deltas)))
    ax3.set_yticklabels(deltas.index, fontsize=9)
    ax3.axvline(0, color="black", lw=0.8, ls="--")
    ax3.set_xlabel("Δ Jaccard (SAE − LogReg)")
    ax3.set_title("Feature reuse advantage per semantic group\n(blue=SAE wins, orange=LogReg wins)")

    legend_patches = [
        plt.matplotlib.patches.Patch(color=SAE_COLOR,    label="SAE more reuse"),
        plt.matplotlib.patches.Patch(color=LOGREG_COLOR, label="LogReg more reuse"),
    ]
    ax3.legend(handles=legend_patches, loc="lower right")

    plt.tight_layout()
    out3 = os.path.join(out_dir, "feature_reuse_delta.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out3}")


if __name__ == "__main__":
    RESULTS_DIR = f"results/interpretability_comparison_{MODEL_NAME}"
    OUT_DIR = RESULTS_DIR

    os.makedirs(OUT_DIR, exist_ok=True)

    sel_df, reuse_df = load_results(RESULTS_DIR)
    print(f"Loaded {len(sel_df)} selectivity rows across {sel_df['dataset'].nunique()} datasets.")
    if not reuse_df.empty:
        print(f"Loaded {len(reuse_df)} feature-reuse pairs across {reuse_df['group'].nunique()} semantic groups.")

    plot_selectivity(sel_df, OUT_DIR)
    plot_feature_reuse(reuse_df, OUT_DIR)
    print(f"\nAll figures saved to: {OUT_DIR}")
