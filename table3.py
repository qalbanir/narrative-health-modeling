"""Table 3: significant racial differences in LIWC-HSI correlations.

Mirrors the layout of the personality-trait Table 3, with the RAND SF-36 scales
replacing the NEO traits. LIWC and HSI variables are z-standardized across the
pooled sample before correlating, so both sides are on a common metric.

z is signed as Fisher z(r_Black) - z(r_White): negative means the association is
stronger in the White sample, positive means stronger in the Black sample.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from hsidata import (
    alpha_by_race,
    HSI_OUTCOMES,
    LIWC_VARS,
    compare_correlations,
    ensure_out_dir,
    load_data,
)

ALPHA = 0.01
SUPPLEMENTARY_ALPHA = 0.05
GROUPS = ["White", "Black"]

LIWC_LABELS = {
    "health": "Health",
    "illness": "Illness",
    "wellness": "Wellness",
    "mental": "Mental health",
    "Physical": "Physical",
    "feeling": "Feeling",
    "fatigue": "Fatigue",
    "emo_neg": "emo_neg",
    "emo_anx": "emo_anx",
    "emo_sad": "emo_sad",
    "emo_pos": "emo_pos",
    "tone_pos": "tone_pos",
}

TABLE_NOTE = (
    "Note. All correlation differences significant at p < {alpha}. LIWC and HSI variables "
    "were z-standardized prior to analysis. z = Fisher z(r_Black) - z(r_White), signed on the "
    "raw correlation scale: positive z means the correlation is more positive (or less "
    "negative) in the Black sample. Higher HSI scores indicate better health. "
    "emo_neg = Negative emotion, emo_anx = Anxiety, emo_sad = Sadness, "
    "emo_pos = Positive emotion, tone_pos = Positive tone."
)

FULL_NOTE = (
    "Note. N = 900 (450 Black, 450 White) propensity-matched participants; cell Ns vary "
    "slightly with item-level missingness. p_White and p_Black test each within-group "
    "correlation against zero; p tests the "
    "racial difference between them. LIWC and HSI variables were z-standardized prior to "
    "analysis. z = Fisher z(r_Black) - z(r_White), signed on the raw correlation scale: "
    "positive z means the correlation is more positive (or less negative) in the Black sample. "
    "Higher HSI scores indicate better health, so negative correlations mean more of that "
    "language accompanies worse health. Physical Health and Mental Health Composites are the "
    "means of their four constituent subscales. Across the 120 difference tests, none survive "
    "Benjamini-Hochberg correction. emo_neg = Negative emotion, emo_anx = Anxiety, "
    "emo_sad = Sadness, emo_pos = Positive emotion, tone_pos = Positive tone."
)


def standardize(df):
    """z-score the LIWC and HSI variables across the pooled sample."""
    out = df.copy()
    for col in LIWC_VARS + HSI_OUTCOMES:
        values = out[col]
        sd = values.std(ddof=1)
        out[col] = (values - values.mean()) / sd if sd > 0 else np.nan
    return out


def correlations(df):
    samples = {g: df[df["race_name"] == g] for g in GROUPS}
    rows = []
    for outcome in HSI_OUTCOMES:
        for liwc_var in LIWC_VARS:
            clean = {g: samples[g][[liwc_var, outcome]].dropna() for g in GROUPS}
            if any(len(c) < 3 for c in clean.values()):
                continue
            stats = {}
            for g in GROUPS:
                r, p = pearsonr(clean[g][liwc_var], clean[g][outcome])
                stats[g] = (r, p, len(clean[g]))
            rW, pW, nW = stats["White"]
            rB, pB, nB = stats["Black"]
            z, comp_p, q = compare_correlations(rB, nB, rW, nW)
            rows.append(
                {
                    "HSI scale": outcome,
                    "LIWC variable": LIWC_LABELS[liwc_var],
                    "N_White": nW,
                    "r_White": rW,
                    "p_White": pW,
                    "N_Black": nB,
                    "r_Black": rB,
                    "p_Black": pB,
                    "z": z,
                    "p": comp_p,
                    "cohens_q": q,
                }
            )
    return pd.DataFrame(rows)


def fmt_p(value):
    """APA p-value: no leading zero, floored at < .001."""
    if pd.isna(value):
        return ""
    return "< .001" if value < 0.001 else f"{value:.3f}".lstrip("0")


COMPACT_COLUMNS = ["HSI subscale", "LIWC variable", "N_White", "r_White", "N_Black", "r_Black", "z"]

FULL_COLUMNS = [
    "HSI subscale", "LIWC variable", "N_White", "r_White", "p_White",
    "N_Black", "r_Black", "p_Black", "z", "p",
]


def format_table(results, alpha=None, compact=False):
    """APA layout: HSI scale printed once per group. Filters to alpha when given."""
    table = results if alpha is None else results[results["p"] < alpha]
    table = table.copy()
    if table.empty:
        return table

    scale_order = {scale: i for i, scale in enumerate(HSI_OUTCOMES)}
    liwc_order = {LIWC_LABELS[v]: i for i, v in enumerate(LIWC_VARS)}
    table["_scale"] = table["HSI scale"].map(scale_order)
    if alpha is None:
        table["_within"] = table["LIWC variable"].map(liwc_order)
        ascending = [True, True]
    else:
        table["_within"] = -table["z"].abs()
        ascending = [True, True]
    table = table.sort_values(["_scale", "_within"], ascending=ascending)

    for col in ["r_White", "r_Black"]:
        table[col] = table[col].map(lambda v: f"{v:.2f}")
    table["z"] = table["z"].map(lambda v: f"{v:.2f}")
    for src, dest in [("p_White", "p_White"), ("p_Black", "p_Black"), ("p", "p")]:
        table[dest] = table[src].map(fmt_p)
    table["HSI scale"] = table["HSI scale"].mask(table["HSI scale"].duplicated(), "")
    table = table.rename(columns={"HSI scale": "HSI subscale"})

    return table[COMPACT_COLUMNS if compact else FULL_COLUMNS]


RANKED_NOTE = (
    "Note. Rows are ordered by the strength of the LIWC-HSI association in the full sample "
    "(most significant pairing first), not by the size of the racial difference. LIWC and HSI "
    "variables were z-standardized prior to analysis. z = Fisher z(r_Black) - z(r_White), "
    "signed on the raw correlation scale: positive z means the correlation is more positive "
    "(or less negative) in the Black sample. Higher HSI scores indicate better health, so a "
    "negative r means more of that language accompanies worse health. emo_neg = Negative "
    "emotion, emo_anx = Anxiety, emo_sad = Sadness, emo_pos = Positive emotion, "
    "tone_pos = Positive tone."
)


SUBSCALE_NOTE = (
    "Note. One row per HSI scale, addressing where racial differences in language-health "
    "associations emerge and where they do not. alpha_W / alpha_B are Cronbach's alpha within "
    "each racial group and Feldt p tests their equality; unequal reliability attenuates "
    "correlations more in the lower-alpha group and is a measurement-side explanation for an "
    "apparent difference. Sig_W and Sig_B count how many of the 12 LIWC categories correlate "
    "with that scale at p < .01 within each group. Diff_05 counts racial differences at "
    "p < .05. Max|z| is the largest racial difference on that scale and the LIWC category "
    "producing it. Mean|r| is the average absolute correlation across all 12 LIWC categories."
)


def subscale_summary(results, reliability):
    """Per-HSI-scale synthesis: reliability, association density, and where races diverge."""
    alphas = reliability.set_index("hsi_scale")
    rows = []
    for scale in HSI_OUTCOMES:
        block = results[results["HSI scale"] == scale]
        peak = block.loc[block["z"].abs().idxmax()]
        rows.append(
            {
                "HSI subscale": scale,
                "alpha_W": round(alphas.loc[scale, "alpha_White"], 3),
                "alpha_B": round(alphas.loc[scale, "alpha_Black"], 3),
                "Feldt_p": alphas.loc[scale, "feldt_p"],
                "Sig_W": int((block["p_White"] < 0.01).sum()),
                "Sig_B": int((block["p_Black"] < 0.01).sum()),
                "Diff_05": int((block["p"] < 0.05).sum()),
                "Max|z|": round(abs(peak["z"]), 2),
                "Max|z| LIWC": peak["LIWC variable"],
                "Mean|r|_W": round(block["r_White"].abs().mean(), 3),
                "Mean|r|_B": round(block["r_Black"].abs().mean(), 3),
            }
        )
    table = pd.DataFrame(rows)
    table["Feldt_p"] = table["Feldt_p"].map(fmt_p)
    return table


def pooled_stats(df):
    """Full-sample r and p for every LIWC x HSI pairing, used to rank the table."""
    stats = {}
    for liwc_var in LIWC_VARS:
        for outcome in HSI_OUTCOMES:
            clean = df[[liwc_var, outcome]].dropna()
            r, p = pearsonr(clean[liwc_var], clean[outcome])
            stats[(LIWC_LABELS[liwc_var], outcome)] = (r, p)
    return stats


def ranked_table(results, df):
    """The 7-column layout, ordered by full-sample significance rather than by scale."""
    stats = pooled_stats(df)
    table = results.copy()
    table["pooled_r"] = [stats[(l, h)][0] for l, h in zip(table["LIWC variable"], table["HSI scale"])]
    table["pooled_p"] = [stats[(l, h)][1] for l, h in zip(table["LIWC variable"], table["HSI scale"])]
    table = table.sort_values("pooled_p")

    formatted = table.rename(columns={"HSI scale": "HSI subscale"}).copy()
    for col in ["r_White", "r_Black"]:
        formatted[col] = formatted[col].map(lambda v: f"{v:.2f}")
    formatted["z"] = formatted["z"].map(lambda v: f"{v:.2f}")
    return formatted[COMPACT_COLUMNS], table


def to_markdown(table, title, subtitle, note):
    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join([f"**{title}**", f"*{subtitle}*", "", header, sep, *body, "", note])


def main():
    out_dir = ensure_out_dir()
    df = load_data()

    raw = correlations(df)
    std = correlations(standardize(df))

    max_shift = (raw["r_White"] - std["r_White"]).abs().max()
    print(f"\nStandardization check: max |r_raw - r_standardized| = {max_shift:.2e}")
    print("Pearson r is scale-invariant, so z-scoring leaves the correlations unchanged.")

    pd.set_option("display.width", 200, "display.max_columns", 20)

    specs = [
        (ALPHA, "table3", "Table 3",
         "Significant Differences in Correlations between LIWC and Health Status "
         "Inventory Scores Across Races (p < .01)", False),
        (SUPPLEMENTARY_ALPHA, "table3_supplementary", "Table S3",
         "Significant Differences in Correlations between LIWC and Health Status "
         "Inventory Scores Across Races (p < .05)", False),
        (None, "table3_full", "Table 4",
         "Correlations between LIWC Categories and Health Status Inventory Scores "
         "by Race, with Tests of Racial Difference", True),
        (None, "table4_full_with_p", "Table 4 (with p values)",
         "Correlations between LIWC Categories and Health Status Inventory Scores "
         "by Race, with Tests of Racial Difference (All 120 Pairings)", False),
    ]

    for alpha, stem, title, subtitle, compact in specs:
        table = format_table(std, alpha, compact=compact)
        print("\n" + "=" * 118)
        print(f"{title}\n{subtitle}")
        print("=" * 118)
        if table.empty:
            print(f"No differences reached p < {alpha}.")
            continue
        print(table.to_string(index=False))
        note = FULL_NOTE if alpha is None else TABLE_NOTE.format(alpha=f"{alpha:.2f}".lstrip("0"))
        print("\n" + note)

        table.to_csv(f"{out_dir}/{stem}_liwc_hsi_race_differences.csv", index=False)
        with open(f"{out_dir}/{stem}_liwc_hsi_race_differences.md", "w") as fh:
            fh.write(to_markdown(table, title, subtitle, note))

    summary = subscale_summary(std, alpha_by_race())
    title = "Table 2"
    subtitle = (
        "Reliability, Association Density, and Racial Differences by Health Status "
        "Inventory Scale"
    )
    print("\n" + "=" * 118)
    print(f"{title}\n{subtitle}")
    print("=" * 118)
    print(summary.to_string(index=False))
    print("\n" + SUBSCALE_NOTE)
    summary.to_csv(f"{out_dir}/table2_subscale_summary.csv", index=False)
    with open(f"{out_dir}/table2_subscale_summary.md", "w") as fh:
        fh.write(to_markdown(summary, title, subtitle, SUBSCALE_NOTE))

    ranked, ranked_with_stats = ranked_table(std, df)
    title = "Table 5"
    subtitle = (
        "Correlations between LIWC Categories and Health Status Inventory Subscales by Race, "
        "Ranked by Strength of Association in the Full Sample"
    )
    print("\n" + "=" * 118)
    print(f"{title}\n{subtitle}")
    print("=" * 118)
    print(ranked.to_string(index=False))
    print("\n" + RANKED_NOTE)

    ranked.to_csv(f"{out_dir}/table5_liwc_hsi_ranked.csv", index=False)
    ranked_with_stats.round(4).to_csv(f"{out_dir}/table5_liwc_hsi_ranked_with_stats.csv", index=False)
    with open(f"{out_dir}/table5_liwc_hsi_ranked.md", "w") as fh:
        fh.write(to_markdown(ranked, title, subtitle, RANKED_NOTE))

    std.to_csv(f"{out_dir}/liwc_hsi_correlations_standardized.csv", index=False)
    print(f"\nSaved tables and full standardized results to {out_dir}/")


if __name__ == "__main__":
    main()
