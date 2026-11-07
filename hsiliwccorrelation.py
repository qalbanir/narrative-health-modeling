"""LIWC health/affect categories correlated with Health Status Inventory scores,
compared between Black and White participants via Fisher r-to-z.

Parallels liwccorrelation.py's design, but the criterion variables are the RAND
SF-36 subscales/composites rather than the NEO personality traits, and the LIWC
side is restricted to the health- and affect-relevant categories.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from hsidata import (
    HSI_COMPOSITES,
    HSI_OUTCOMES,
    LIWC_VARS,
    alpha_by_race,
    compare_correlations,
    ensure_out_dir,
    load_data,
)

ALPHA = 0.01
GROUPS = ["White", "Black"]


def add_disattenuated(results, reliability):
    """Correct each group's r for criterion unreliability and re-test the difference.

    LIWC frequencies are treated as observed, so only the HSI side is corrected:
    r_true = r_obs / sqrt(alpha). Where alpha is lower in one group, that group's
    correlations are attenuated more, which alone can produce an apparent race
    difference. This is a sensitivity check, not a replacement for the raw test.
    """
    alphas = reliability.set_index("hsi_scale")
    merged = results.merge(
        alphas[["alpha_White", "alpha_Black"]], left_on="hsi_scale", right_index=True, how="left"
    )

    for group in GROUPS:
        key = group.lower()
        merged[f"{key}_r_dis"] = merged[f"{key}_r"] / np.sqrt(merged[f"alpha_{group}"])
    merged[["black_r_dis", "white_r_dis"]] = merged[["black_r_dis", "white_r_dis"]].clip(-0.999, 0.999)

    dis = merged.apply(
        lambda row: compare_correlations(
            row["black_r_dis"], row["black_n"], row["white_r_dis"], row["white_n"]
        ),
        axis=1,
        result_type="expand",
    )
    merged[["z_dis", "comparison_p_dis", "cohens_q_dis"]] = dis
    return merged


def run_correlations(df):
    samples = {g: df[df["race_name"] == g] for g in GROUPS}
    rows = []

    for liwc_var in LIWC_VARS:
        for outcome in HSI_OUTCOMES:
            clean = {g: samples[g][[liwc_var, outcome]].dropna() for g in GROUPS}
            if any(len(c) < 3 for c in clean.values()):
                print(f"  Skipping {liwc_var} x {outcome}: insufficient data")
                continue

            stats_by_group = {}
            for g in GROUPS:
                r, p = pearsonr(clean[g][liwc_var], clean[g][outcome])
                stats_by_group[g] = (r, p, len(clean[g]))

            rW, pW, nW = stats_by_group["White"]
            rB, pB, nB = stats_by_group["Black"]
            z, comp_p, q = compare_correlations(rB, nB, rW, nW)

            rows.append(
                {
                    "liwc_variable": liwc_var,
                    "hsi_scale": outcome,
                    "hsi_type": "Composite" if outcome in HSI_COMPOSITES else "Subscale",
                    "black_r": rB,
                    "black_p": pB,
                    "black_n": nB,
                    "white_r": rW,
                    "white_p": pW,
                    "white_n": nW,
                    "z_statistic": z,
                    "comparison_p": comp_p,
                    "cohens_q": q,
                }
            )

    results = pd.DataFrame(rows)
    for col in ["black_p", "white_p", "comparison_p"]:
        results[col.replace("_p", "_q_fdr")] = multipletests(results[col], method="fdr_bh")[1]
    return results


def pooled_matrix(df):
    """Full-sample LIWC x HSI correlations, collapsing across race."""
    r_rows, p_rows = {}, {}
    for liwc_var in LIWC_VARS:
        r_rows[liwc_var], p_rows[liwc_var] = {}, {}
        for outcome in HSI_OUTCOMES:
            clean = df[[liwc_var, outcome]].dropna()
            r, p = pearsonr(clean[liwc_var], clean[outcome])
            r_rows[liwc_var][outcome] = r
            p_rows[liwc_var][outcome] = p
    r_mat = pd.DataFrame(r_rows).T.reindex(index=LIWC_VARS, columns=HSI_OUTCOMES)
    p_mat = pd.DataFrame(p_rows).T.reindex(index=LIWC_VARS, columns=HSI_OUTCOMES)
    return r_mat, p_mat


def starred(r_mat, p_mat):
    """r with APA significance markers."""
    def mark(r, p):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return f"{r:.2f}{stars}"

    return pd.DataFrame(
        {col: [mark(r_mat.loc[i, col], p_mat.loc[i, col]) for i in r_mat.index] for col in r_mat.columns},
        index=r_mat.index,
    )


def report(results):
    def show(frame, cols):
        print(frame[cols].round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("ALL CORRELATIONS: LIWC category x HSI scale, by race")
    print("=" * 100)
    show(
        results,
        ["liwc_variable", "hsi_scale", "black_r", "black_p", "white_r", "white_p",
         "z_statistic", "comparison_p"],
    )

    print("\n" + "=" * 100)
    print(f"RACIAL DIFFERENCES IN CORRELATION MAGNITUDE (p < {ALPHA})")
    print("=" * 100)
    diff = results[results["comparison_p"] < ALPHA].sort_values("comparison_p")
    if diff.empty:
        print(f"No significant differences at p < {ALPHA}.")
    else:
        show(diff, ["liwc_variable", "hsi_scale", "black_r", "white_r", "z_statistic",
                    "comparison_p", "comparison_q_fdr", "cohens_q"])
        print("\nSensitivity check, corrected for unequal criterion reliability:")
        show(diff, ["liwc_variable", "hsi_scale", "alpha_White", "alpha_Black",
                    "black_r_dis", "white_r_dis", "z_dis", "comparison_p_dis"])
        survived = diff[diff["comparison_p_dis"] < ALPHA]
        print(f"\n{len(survived)} of {len(diff)} survive disattenuation at p < {ALPHA}"
              f"{'.' if survived.empty else ': ' + ', '.join(survived['liwc_variable'] + ' x ' + survived['hsi_scale'])}")

    for group, r_col, p_col, q_col in [
        ("BLACK", "black_r", "black_p", "black_q_fdr"),
        ("WHITE", "white_r", "white_p", "white_q_fdr"),
    ]:
        print("\n" + "=" * 100)
        print(f"SIGNIFICANT CORRELATIONS IN {group} SAMPLE (p < {ALPHA})")
        print("=" * 100)
        sig = results[results[p_col] < ALPHA].copy()
        if sig.empty:
            print(f"None at p < {ALPHA}.")
            continue
        sig["abs_r"] = sig[r_col].abs()
        sig = sig.sort_values("abs_r", ascending=False)
        show(sig, ["liwc_variable", "hsi_scale", r_col, p_col, q_col])

    print("\n" + "=" * 100)
    print(f"SIGNIFICANT IN BOTH SAMPLES (p < {ALPHA})")
    print("=" * 100)
    both = results[(results["black_p"] < ALPHA) & (results["white_p"] < ALPHA)]
    if both.empty:
        print(f"None at p < {ALPHA}.")
    else:
        show(both, ["liwc_variable", "hsi_scale", "black_r", "white_r", "comparison_p"])

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    n = len(results)
    print(f"LIWC categories: {len(LIWC_VARS)} | HSI scales: {len(HSI_OUTCOMES)} | Tests: {n}")
    print(f"Black sample:      p < {ALPHA}: {(results['black_p'] < ALPHA).sum():3d}"
          f" | BH q < .05: {(results['black_q_fdr'] < .05).sum():3d}")
    print(f"White sample:      p < {ALPHA}: {(results['white_p'] < ALPHA).sum():3d}"
          f" | BH q < .05: {(results['white_q_fdr'] < .05).sum():3d}")
    print(f"Race differences:  p < {ALPHA}: {(results['comparison_p'] < ALPHA).sum():3d}"
          f" | BH q < .05: {(results['comparison_q_fdr'] < .05).sum():3d}")


def main():
    out_dir = ensure_out_dir()
    df = load_data()
    results = run_correlations(df)
    results = add_disattenuated(results, alpha_by_race())

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    report(results)

    r_mat, p_mat = pooled_matrix(df)
    print("\n" + "=" * 100)
    print("POOLED CORRELATION MATRIX: LIWC category x HSI scale (full sample, N = 900)")
    print("* p < .05, ** p < .01, *** p < .001")
    print("=" * 100)
    print(starred(r_mat, p_mat).to_string())

    path = f"{out_dir}/liwc_hsi_correlations_by_race.csv"
    results.to_csv(path, index=False)
    starred(r_mat, p_mat).to_csv(f"{out_dir}/liwc_hsi_correlation_matrix_pooled.csv")
    r_mat.round(3).to_csv(f"{out_dir}/liwc_hsi_correlation_matrix_pooled_r.csv")
    p_mat.round(4).to_csv(f"{out_dir}/liwc_hsi_correlation_matrix_pooled_p.csv")

    matrix = results.pivot(index="liwc_variable", columns="hsi_scale", values="black_r").round(3)
    matrix_path = f"{out_dir}/liwc_hsi_correlation_matrix_black.csv"
    matrix.to_csv(matrix_path)
    white_matrix = results.pivot(index="liwc_variable", columns="hsi_scale", values="white_r").round(3)
    white_path = f"{out_dir}/liwc_hsi_correlation_matrix_white.csv"
    white_matrix.to_csv(white_path)

    print(f"\nSaved:\n  {path}\n  {matrix_path}\n  {white_path}")


if __name__ == "__main__":
    main()
