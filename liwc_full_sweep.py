"""Does the null result for racial differences in LIWC-health associations hold
across the whole LIWC space, or only across the twelve categories selected a priori?

The main analysis tests 12 theory-selected categories against 10 HSI outcomes (120
comparisons) and finds no racial differences surviving correction. A sceptic can
reasonably ask whether those twelve were the categories that happened to show
nothing. This sweeps every numeric LIWC variable in the file against all ten
outcomes and applies the same Fisher r-to-z test and Benjamini-Hochberg correction.

A null that survives the full sweep is much stronger evidence than a null over a
hand-picked subset. A null that does not survive tells you the selection mattered,
which is also worth knowing.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from hsidata import HSI_OUTCOMES, LIWC_PATH, LIWC_VARS, compare_correlations, ensure_out_dir, load_data

OUT = ensure_out_dir()
# Identifiers and LIWC bookkeeping columns, not psychological categories.
NON_CATEGORIES = {"PARTID", "PRACE", "Segment", "WC"}
MIN_N = 50


def liwc_categories(df):
    cats = [c for c in df.columns
            if c not in NON_CATEGORIES
            and c not in HSI_OUTCOMES
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().sum() > MIN_N
            and df[c].nunique() > 2]
    return cats


def main():
    data = load_data()
    liwc = pd.read_csv(LIWC_PATH, low_memory=False)
    cats = liwc_categories(liwc)
    keep = [c for c in cats if c not in data.columns]
    merged = data.merge(liwc[["PARTID"] + keep], on="PARTID", how="left") if keep else data
    cats = [c for c in cats if c in merged.columns]
    print(f"{len(cats)} LIWC categories x {len(HSI_OUTCOMES)} outcomes "
          f"= {len(cats) * len(HSI_OUTCOMES)} comparisons\n")

    rows = []
    for cat in cats:
        for outcome in HSI_OUTCOMES:
            sub = merged.dropna(subset=[cat, outcome, "race_name"])
            w = sub[sub["race_name"] == "White"]
            b = sub[sub["race_name"] == "Black"]
            if len(w) < MIN_N or len(b) < MIN_N:
                continue
            rw = w[cat].corr(w[outcome])
            rb = b[cat].corr(b[outcome])
            if not np.isfinite(rw) or not np.isfinite(rb):
                continue
            z, pval, cohens_q = compare_correlations(rw, len(w), rb, len(b))
            rows.append({"liwc_variable": cat, "hsi_outcome": outcome,
                         "selected_12": cat in LIWC_VARS,
                         "n_White": len(w), "n_Black": len(b),
                         "r_White": round(rw, 3), "r_Black": round(rb, 3),
                         "z": round(z, 3), "p": pval, "cohens_q": round(cohens_q, 3)})

    out = pd.DataFrame(rows)
    out["q_BH"] = multipletests(out["p"], method="fdr_bh")[1]
    n = len(out)

    print("=" * 76)
    print("FULL LIWC SWEEP: racial differences in LIWC-HSI associations")
    print("=" * 76)
    print(f"  comparisons tested          {n}")
    print(f"  expected false positives    {n * 0.05:.0f}   (at alpha = .05)")
    print(f"  observed p < .05            {(out['p'] < .05).sum()}")
    print(f"  observed p < .01            {(out['p'] < .01).sum()}")
    print(f"  surviving BH (q < .05)      {(out['q_BH'] < .05).sum()}")

    sel = out[out["selected_12"]]
    print(f"\n  within the 12 selected categories ({len(sel)} comparisons):")
    print(f"    p < .05                   {(sel['p'] < .05).sum()}  "
          f"(expected {len(sel) * 0.05:.0f})")
    print(f"    surviving BH              {(sel['q_BH'] < .05).sum()}")

    surv = out[out["q_BH"] < .05]
    if len(surv):
        print("\n  comparisons surviving correction:")
        print(surv.sort_values("q_BH").head(20).to_string(index=False))
    else:
        print("\n  No comparison survived correction anywhere in the LIWC space.")
        top = out.nsmallest(5, "p")[["liwc_variable", "hsi_outcome", "r_White", "r_Black", "z", "p", "q_BH"]]
        print("\n  five smallest uncorrected p values, for reference:")
        print(top.to_string(index=False))

    out.to_csv(f"{OUT}/liwc_full_sweep.csv", index=False)
    print(f"\nSaved: {OUT}/liwc_full_sweep.csv")


if __name__ == "__main__":
    main()
