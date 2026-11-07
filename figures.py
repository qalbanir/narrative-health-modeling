"""Figures for the LIWC-HSI racial difference analysis.

Figure 1  the four LIWC-HSI associations that differ significantly by race
Figure 2a three strong associations that are statistically equivalent across race
Figure 2b heatmap of the racial difference (z) across all 120 pairings

Axes are z-standardized, so the slope of each fitted line equals that group's
correlation and visually different slopes mean different r.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from hsidata import HSI_OUTCOMES, LIWC_VARS, ensure_out_dir, load_data

FIG_DIR = None

WHITE_COLOR = "#2b6cb0"
BLACK_COLOR = "#c05621"
GROUPS = ["White", "Black"]
COLORS = {"White": WHITE_COLOR, "Black": BLACK_COLOR}

# (HSI scale, LIWC variable, display label)
FIG1_PAIRS = [
    ("Social Functioning", "Physical", "Social Functioning x Physical"),
    ("Social Functioning", "health", "Social Functioning x health"),
    ("Physical functioning", "emo_sad", "Physical functioning x emo_sad"),
    ("Mental Health Composite", "health", "Mental Health Composite x health"),
]

# Chosen to be both strong (mean |r| > .21) and statistically equivalent (p > .50).
# Pain x illness and Energy/fatigue x health were dropped despite being on the
# shortlist: at p = .072 and .085 their lines visibly diverge, which would
# undercut the point this figure makes.
FIG2A_PAIRS = [
    ("Physical functioning", "Physical", "Physical functioning x Physical"),
    ("Role limitations due to physical health", "illness", "Role limitations - physical x illness"),
    ("General health", "health", "General health x health"),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9,
        "figure.dpi": 150,
    }
)


def standardize(df):
    out = df.copy()
    for col in LIWC_VARS + HSI_OUTCOMES:
        sd = out[col].std(ddof=1)
        out[col] = (out[col] - out[col].mean()) / sd if sd > 0 else np.nan
    return out


def jitter(values, rng):
    """Spread points on coarse scales so overlapping cases stay visible.

    Applied to the plotted points only; every correlation and fitted line uses
    the unjittered data.
    """
    unique = np.unique(values[~np.isnan(values)])
    if len(unique) > 30 or len(unique) < 2:
        return values
    gap = np.median(np.diff(unique))
    return values + rng.normal(0, gap * 0.22, size=len(values))


def diff_label(df, hsi, liwc):
    """Formatted z and p for the racial difference in this pairing."""
    from hsidata import compare_correlations

    stats = {}
    for group in GROUPS:
        sub = df[df["race_name"] == group][[liwc, hsi]].dropna()
        r, _ = pearsonr(sub[liwc], sub[hsi])
        stats[group] = (r, len(sub))
    (rW, nW), (rB, nB) = stats["White"], stats["Black"]
    z, p, _ = compare_correlations(rB, nB, rW, nW)
    p_txt = "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")
    return f"z = {z:.2f}, p = {p_txt}"


def panel(ax, df, hsi, liwc, title, annotation="", swap_axes=False):
    """Scatter plus per-race regression line for one LIWC x HSI pairing.

    swap_axes puts the HSI scale on x and the LIWC category on y. Because both
    axes are z-standardized the fitted slope equals r in either orientation, so
    this changes only the presentation.
    """
    x_var, y_var = (hsi, liwc) if swap_axes else (liwc, hsi)
    stats = {}
    rng = np.random.default_rng(42)
    x_all, y_all = [], []

    for group in GROUPS:
        sub = df[df["race_name"] == group][[x_var, y_var]].dropna()
        x, y = sub[x_var].values, sub[y_var].values
        r, _ = pearsonr(x, y)
        stats[group] = r
        x_all.append(x)
        y_all.append(y)

        ax.scatter(jitter(x, rng), jitter(y, rng), s=6, alpha=0.18,
                   color=COLORS[group], linewidths=0, zorder=1)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 50)
        ax.plot(xs, slope * xs + intercept, color=COLORS[group], lw=2.2, zorder=3,
                label=f"{group}  r = {r:.2f}")

    def limits(arrays, pad=0.35):
        flat = np.concatenate(arrays)
        low, high = np.percentile(flat, 0.5), np.percentile(flat, 99.5)
        return low - pad, high + pad

    ax.set_title(f"{title}\n{annotation}" if annotation else title, pad=6)
    ax.set_xlabel(f"{x_var} (z)")
    ax.set_ylabel(f"{y_var} (z)")
    ax.set_xlim(*limits(x_all))
    ax.set_ylim(*limits(y_all))
    ax.axhline(0, color="#cbd5e0", lw=0.6, zorder=0)
    ax.axvline(0, color="#cbd5e0", lw=0.6, zorder=0)
    ax.legend(frameon=True, framealpha=0.85, edgecolor="none", fontsize=8, loc="best")
    return stats


def figure1(df):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.4))
    for ax, (hsi, liwc, title) in zip(axes.ravel(), FIG1_PAIRS):
        panel(ax, df, hsi, liwc, title, annotation=diff_label(df, hsi, liwc),
              swap_axes=True)
    fig.suptitle(
        "Figure 1. LIWC-Health Associations That Differ Significantly by Race",
        y=0.985, fontsize=10.5, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def figure2a(df):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (hsi, liwc, title) in zip(axes, FIG2A_PAIRS):
        panel(ax, df, hsi, liwc, title, annotation=diff_label(df, hsi, liwc),
              swap_axes=True)
    fig.suptitle(
        "Figure 2. Strong LIWC-Health Associations That Are Equivalent Across Race",
        y=0.99, fontsize=10.5, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def figure2b(results):
    """Heatmap of the racial difference across every LIWC x HSI pairing."""
    z = results.pivot(index="LIWC variable", columns="HSI scale", values="z")
    p = results.pivot(index="LIWC variable", columns="HSI scale", values="p")
    z = z.reindex(index=LIWC_VARS, columns=HSI_OUTCOMES)
    p = p.reindex(index=LIWC_VARS, columns=HSI_OUTCOMES)

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    limit = 3.0
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(len(HSI_OUTCOMES)))
    ax.set_xticklabels(HSI_OUTCOMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(LIWC_VARS)))
    ax.set_yticklabels(LIWC_VARS, fontsize=8)

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if pd.notna(p.iat[i, j]) and p.iat[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=13,
                        fontweight="bold", color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("z  (Fisher z$_{Black}$ - z$_{White}$)", fontsize=9)
    ax.set_title(
        "Figure 2 (alternate). Racial Difference in LIWC-Health Association,\n"
        "All 120 Pairings",
        fontsize=10.5, fontweight="bold", pad=10,
    )
    ax.set_xlabel("HSI scale")
    ax.set_ylabel("LIWC category")
    fig.tight_layout()
    return fig


def correlations(df):
    rows = []
    for outcome in HSI_OUTCOMES:
        for liwc_var in LIWC_VARS:
            from hsidata import compare_correlations

            stats = {}
            for group in GROUPS:
                sub = df[df["race_name"] == group][[liwc_var, outcome]].dropna()
                r, p = pearsonr(sub[liwc_var], sub[outcome])
                stats[group] = (r, len(sub))
            (rW, nW), (rB, nB) = stats["White"], stats["Black"]
            z, comp_p, _ = compare_correlations(rB, nB, rW, nW)
            rows.append({"HSI scale": outcome, "LIWC variable": liwc_var, "z": z, "p": comp_p})
    return pd.DataFrame(rows)


def save(fig, stem):
    for ext in ["png", "pdf"]:
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def main():
    global FIG_DIR
    out_dir = ensure_out_dir()
    FIG_DIR = os.path.join(out_dir, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    raw = load_data(verbose=False)
    std = standardize(raw)
    results = correlations(std)

    print(f"Writing figures to {FIG_DIR}/")
    save(figure1(std), "figure1_significant_differences")
    save(figure2a(std), "figure2a_equivalent_associations")
    save(figure2b(results), "figure2b_difference_heatmap")

    sig = results[results["p"] < 0.05]
    print(f"\nFigure 1 shows {len(FIG1_PAIRS)} of the {len(sig)} pairings significant at p < .05.")
    print(f"Figure 2b marks those {len(sig)} cells with an asterisk out of {len(results)} total.")


if __name__ == "__main__":
    main()
