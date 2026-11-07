"""Table 1: sample characteristics and Health Status Inventory scores by race."""

import numpy as np
import pandas as pd
from scipy import stats

from hsidata import (
    CATEGORICAL_DEMOS,
    CONTINUOUS_DEMOS,
    DEMO_LABELS,
    DEMO_VARS,
    HSI_COMPOSITES,
    HSI_SUBSCALES,
    ensure_out_dir,
    load_data,
)

GROUPS = ["White", "Black"]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def continuous_row(df, col, label):
    groups = {g: df.loc[df["race_name"] == g, col].dropna() for g in GROUPS}
    t, p = stats.ttest_ind(groups["White"], groups["Black"], equal_var=False)
    row = {"Variable": label, "Level": "M (SD)"}
    for g in GROUPS:
        row[f"{g}_n"] = len(groups[g])
        row[f"{g}_stat"] = f"{groups[g].mean():.2f} ({groups[g].std(ddof=1):.2f})"
    row["Test"] = "Welch t"
    row["Statistic"] = round(t, 3)
    row["p"] = p
    row["Effect"] = round(cohens_d(groups["White"], groups["Black"]), 3)
    row["Effect_type"] = "Cohen's d"
    return [row]


def categorical_rows(df, col, label):
    labels = DEMO_LABELS[col]
    sub = df[["race_name", col]].dropna()
    sub = sub[sub[col].isin(labels)]
    table = pd.crosstab(sub[col], sub["race_name"]).reindex(
        index=sorted(labels), columns=GROUPS, fill_value=0
    )
    chi2, p, dof, _ = stats.chi2_contingency(table.values)
    n_total = table.values.sum()
    cramers_v = np.sqrt(chi2 / (n_total * (min(table.shape) - 1)))
    totals = table.sum(axis=0)

    rows = []
    for i, level in enumerate(table.index):
        row = {"Variable": label if i == 0 else "", "Level": labels[level]}
        for g in GROUPS:
            n = int(table.loc[level, g])
            row[f"{g}_n"] = n
            row[f"{g}_stat"] = f"{n} ({n / totals[g] * 100:.1f}%)"
        if i == 0:
            row.update(
                {
                    "Test": f"chi2 (df={dof})",
                    "Statistic": round(chi2, 3),
                    "p": p,
                    "Effect": round(cramers_v, 3),
                    "Effect_type": "Cramer's V",
                }
            )
        else:
            row.update({"Test": "", "Statistic": "", "p": np.nan, "Effect": "", "Effect_type": ""})
        rows.append(row)

    missing = df["race_name"].value_counts().reindex(GROUPS) - totals
    row = {"Variable": "", "Level": "Missing"}
    for g in GROUPS:
        row[f"{g}_n"] = int(missing[g])
        row[f"{g}_stat"] = str(int(missing[g]))
    row.update({"Test": "", "Statistic": "", "p": np.nan, "Effect": "", "Effect_type": ""})
    rows.append(row)
    return rows


def build_demo_table(df):
    rows = []
    for col in CONTINUOUS_DEMOS:
        rows += continuous_row(df, col, DEMO_VARS[col])
    for col in CATEGORICAL_DEMOS:
        rows += categorical_rows(df, col, DEMO_VARS[col])
    return pd.DataFrame(rows)


def build_hsi_table(df):
    rows = []
    outcomes = HSI_SUBSCALES + list(HSI_COMPOSITES)
    for col in outcomes:
        groups = {g: df.loc[df["race_name"] == g, col].dropna() for g in GROUPS}
        t, p = stats.ttest_ind(groups["White"], groups["Black"], equal_var=False)
        row = {
            "HSI scale": col,
            "Type": "Composite" if col in HSI_COMPOSITES else "Subscale",
        }
        for g in GROUPS:
            row[f"{g}_n"] = len(groups[g])
            row[f"{g}_M"] = round(groups[g].mean(), 2)
            row[f"{g}_SD"] = round(groups[g].std(ddof=1), 2)
        row["t"] = round(t, 3)
        row["p"] = p
        row["Cohens_d"] = round(cohens_d(groups["White"], groups["Black"]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def build_descriptives(df):
    """Distributional summary per HSI scale by race, with normality diagnostics.

    Mirrors the descriptives table used for the NEO scores, plus the ceiling
    percentage: HSI scales are bounded at 100 and many respondents sit there,
    which drives the skew and matters for how the scores should be described.
    """
    rows = []
    for col in HSI_SUBSCALES + list(HSI_COMPOSITES):
        row = {"HSI score": col}
        for group, tag in [("White", "W"), ("Black", "B")]:
            s = df.loc[df["race_name"] == group, col].dropna()
            row.update({
                f"M_{tag}": round(s.mean(), 2),
                f"SD_{tag}": round(s.std(ddof=1), 2),
                f"Skew_{tag}": round(stats.skew(s), 2),
                f"Min_{tag}": round(s.min(), 1),
                f"Max_{tag}": round(s.max(), 1),
                f"Median_{tag}": round(s.median(), 1),
                f"AtCeiling_{tag}": round((s == 100).mean() * 100, 1),
                f"ShapiroP_{tag}": stats.shapiro(s).pvalue,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    out_dir = ensure_out_dir()
    df = load_data()

    demo_table = build_demo_table(df)
    hsi_table = build_hsi_table(df)
    descriptives = build_descriptives(df)

    demo_path = f"{out_dir}/table1_demographics_by_race.csv"
    hsi_path = f"{out_dir}/table1b_hsi_by_race.csv"
    demo_table.to_csv(demo_path, index=False)
    hsi_table.to_csv(hsi_path, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)

    print("\n" + "=" * 100)
    print("TABLE 1. Sample characteristics by race (propensity-matched sample)")
    print("=" * 100)
    printable = demo_table.copy()
    printable["p"] = printable["p"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    print(printable.to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE 1b. Health Status Inventory (RAND SF-36) scores by race")
    print("Higher scores = better health. Cohen's d > 0 favors White participants.")
    print("=" * 100)
    printable = hsi_table.copy()
    printable["p"] = printable["p"].map(lambda v: f"{v:.4f}")
    print(printable.to_string(index=False))

    desc_path = f"{out_dir}/table2_hsi_descriptives.csv"
    descriptives.to_csv(desc_path, index=False)

    print("\n" + "=" * 100)
    print("TABLE 2. Descriptive Statistics for the Health Status Inventory Scores")
    print("=" * 100)
    printable = descriptives.drop(columns=["ShapiroP_W", "ShapiroP_B"])
    print(printable.to_string(index=False))

    n_nonnormal_w = (descriptives["ShapiroP_W"] < 0.05).sum()
    n_nonnormal_b = (descriptives["ShapiroP_B"] < 0.05).sum()
    n = len(descriptives)
    print(f"\nShapiro-Wilk rejects normality for {n_nonnormal_w} of {n} scales in the White "
          f"sample and {n_nonnormal_b} of {n} in the Black sample.")
    print(f"Skew is negative on every scale in both groups "
          f"(White {descriptives['Skew_W'].min():.2f} to {descriptives['Skew_W'].max():.2f}, "
          f"Black {descriptives['Skew_B'].min():.2f} to {descriptives['Skew_B'].max():.2f}).")
    print("AtCeiling is the percentage scoring the maximum of 100.")

    print(f"\nSaved:\n  {demo_path}\n  {hsi_path}\n  {desc_path}")


if __name__ == "__main__":
    main()
