"""Why did personality modelling show a racial disparity when health modelling did not?

One candidate explanation is testable with the data already here. If a construct is
predicted from the same linguistic features that mark race, a model of that
construct is drawing on a feature space entangled with group membership. If it is
predicted from features orthogonal to race, it is not.

For each LIWC category this computes two quantities:

    predictive power    |r| with the outcome, pooled across groups
    race-marking power  |Cohen's d| between White and Black narratives

and correlates them across categories. A high correlation means the features that
predict the outcome are the same ones that identify race.

Run for all five FFM traits and all ten HSI outcomes, so the two papers'
constructs can be compared on the same scale.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from hsidata import HSI_OUTCOMES, LIWC_PATH, RACE_MAP, ensure_out_dir, load_data

OUT = ensure_out_dir()
TRAITS = {"PNEON_scaled": "Neuroticism", "PNEOE_scaled": "Extraversion",
          "PNEOO_scaled": "Openness", "PNEOA_scaled": "Agreeableness",
          "PNEOC_scaled": "Conscientiousness"}
NON_CATEGORIES = {"PARTID", "PRACE", "Segment", "WC"}


def cohens_d(a, b):
    sa, sb = a.std(), b.std()
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled else np.nan


def main():
    liwc = pd.read_csv(LIWC_PATH, low_memory=False)
    # FFM scores are stored as text with "." for missing, as PGENDER is.
    for col in TRAITS:
        if col in liwc.columns:
            liwc[col] = pd.to_numeric(liwc[col], errors="coerce")
    liwc["race_name"] = liwc["PRACE"].map(RACE_MAP)
    liwc = liwc[liwc["race_name"].isin(["White", "Black"])]

    cats = [c for c in liwc.columns
            if c not in NON_CATEGORIES and c not in TRAITS and c != "race_name"
            and pd.api.types.is_numeric_dtype(liwc[c])
            and liwc[c].nunique() > 2 and liwc[c].notna().sum() > 100]

    # race-marking power of each category
    w, b = liwc[liwc["race_name"] == "White"], liwc[liwc["race_name"] == "Black"]
    race_d = pd.Series({c: abs(cohens_d(w[c].dropna(), b[c].dropna())) for c in cats})

    # HSI outcomes live in the scored file, not the LIWC export
    hsi = load_data()[["PARTID"] + HSI_OUTCOMES]
    merged = liwc.merge(hsi, on="PARTID", how="left")

    rows = []
    for col, label in list(TRAITS.items()) + [(o, o) for o in HSI_OUTCOMES]:
        if col not in merged.columns:
            continue
        pred = pd.Series({c: abs(merged[c].corr(merged[col])) for c in cats}).dropna()
        shared = pred.index.intersection(race_d.index)
        r, p = pearsonr(pred[shared], race_d[shared])
        rows.append({"construct": label,
                     "kind": "FFM trait" if col in TRAITS else "HSI outcome",
                     "n_categories": len(shared),
                     "r(predictive power, race-marking power)": round(r, 3),
                     "p": p})

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 150)
    print("=" * 94)
    print("OVERLAP BETWEEN PREDICTIVE FEATURES AND RACE-MARKING FEATURES")
    print("=" * 94)
    print(out.assign(p=out["p"].map(lambda v: "<.001" if v < .001 else f"{v:.3f}"))
          .to_string(index=False))

    for kind in ["FFM trait", "HSI outcome"]:
        s = out[out["kind"] == kind]["r(predictive power, race-marking power)"]
        print(f"\n{kind:12s}  mean r = {s.mean():.3f}   range {s.min():.3f} to {s.max():.3f}")

    out.to_csv(f"{OUT}/feature_overlap_race.csv", index=False)
    print(f"\nSaved: {OUT}/feature_overlap_race.csv")


if __name__ == "__main__":
    main()
