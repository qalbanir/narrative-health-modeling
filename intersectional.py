"""Race x gender analysis of HSI scores and of model bias.

Two questions the main analyses cannot answer:

  1. Does the racial gap in self-reported health differ for men and women?
  2. Does the model's directional offset differ for men and women?

The second reuses the pooled held-out predictions already written by the fine-tuned
models, so nothing is refit. Gender comes from the demographics file; race from the
same source the main analyses use.

A warning that belongs in any write-up of this: detecting an interaction needs
roughly four times the sample that detecting a main effect of the same size does.
With about 200 participants per race-by-gender cell, this analysis is well powered
for the cell descriptives and underpowered for the interactions. Null interactions
here are not evidence of no interaction.
"""

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from bias_analyses import load_race, pooled
from hsidata import DEMO_PATH, HSI_OUTCOMES, ensure_out_dir, load_data
from rq2_results import OUTCOMES

OUT = ensure_out_dir()
GENDER = {1: "Male", 2: "Female"}


def with_gender(df):
    """PGENDER is stored as text ('1', '2', '.'), so coerce before mapping."""
    g = pd.read_csv(DEMO_PATH, low_memory=False, usecols=["PARTID", "PGENDER"])
    g["gender"] = pd.to_numeric(g["PGENDER"], errors="coerce").map(GENDER)
    merged = df.merge(g[["PARTID", "gender"]], on="PARTID", how="left")
    if merged["gender"].notna().sum() == 0:
        raise SystemExit("Gender merge produced no matches; check PARTID and PGENDER coding.")
    return merged


def cell_sizes(df):
    t = pd.crosstab(df["race_name"], df["gender"])
    print("CELL SIZES")
    print(t.to_string(), "\n")
    return t


def hsi_by_cell(df):
    """Observed HSI scores in each race x gender cell, with a 2x2 model per outcome."""
    rows = []
    for outcome in HSI_OUTCOMES:
        sub = df.dropna(subset=[outcome, "gender", "race_name"]).copy()
        sub["black"] = (sub["race_name"] == "Black").astype(int)
        sub["female"] = (sub["gender"] == "Female").astype(int)
        fit = smf.ols(f'Q("{outcome}") ~ black * female', data=sub).fit()
        row = {"HSI outcome": outcome}
        for r in ["White", "Black"]:
            for g in ["Male", "Female"]:
                cell = sub[(sub["race_name"] == r) & (sub["gender"] == g)][outcome]
                row[f"{r[0]}_{g[0]}"] = round(cell.mean(), 1)
        row.update({
            "b_race": round(fit.params["black"], 2), "p_race": fit.pvalues["black"],
            "b_gender": round(fit.params["female"], 2), "p_gender": fit.pvalues["female"],
            "b_interaction": round(fit.params["black:female"], 2),
            "p_interaction": fit.pvalues["black:female"],
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ["race", "gender", "interaction"]:
        out[f"q_{col}"] = multipletests(out[f"p_{col}"], method="fdr_bh")[1]
    return out


def model_bias_by_cell(race):
    """Accuracy per cell, and the race x gender interaction in directional bias."""
    acc_rows, dir_rows = [], []
    for stem, outcome in OUTCOMES.items():
        df = pooled(stem, race)
        if df is None:
            continue
        df = with_gender(df).dropna(subset=["gender"])

        acc = {"HSI outcome": outcome}
        for r in ["White", "Black"]:
            for g in ["Male", "Female"]:
                c = df[(df["race_name"] == r) & (df["gender"] == g)]
                acc[f"n_{r[0]}{g[0]}"] = len(c)
                acc[f"r_{r[0]}{g[0]}"] = round(pearsonr(c["true_label"], c["predicted_label"])[0], 3) \
                    if len(c) > 10 else None
        acc_rows.append(acc)

        df["female"] = (df["gender"] == "Female").astype(int)
        fit = smf.ols("error ~ black * female + true_label", data=df).fit()
        dir_rows.append({
            "HSI outcome": outcome,
            "b_race": round(fit.params["black"], 3), "p_race": fit.pvalues["black"],
            "b_gender": round(fit.params["female"], 3), "p_gender": fit.pvalues["female"],
            "b_interaction": round(fit.params["black:female"], 3),
            "SE_int": round(fit.bse["black:female"], 3),
            "p_interaction": fit.pvalues["black:female"],
        })

    if not dir_rows:
        return None, None
    acc = pd.DataFrame(acc_rows)
    direc = pd.DataFrame(dir_rows)
    for col in ["race", "gender", "interaction"]:
        direc[f"q_{col}"] = multipletests(direc[f"p_{col}"], method="fdr_bh")[1]
    return acc, direc


def show(title, frame, pcols):
    out = frame.copy()
    for c in pcols:
        out[c] = out[c].map(lambda v: "<.001" if v < .001 else f"{v:.3f}".lstrip("0"))
    print("=" * 110)
    print(title)
    print("=" * 110)
    print(out.to_string(index=False), "\n")


def main():
    pd.set_option("display.width", 230, "display.max_columns", 30)

    scores = with_gender(load_data()).dropna(subset=["gender"])
    cell_sizes(scores)

    hsi = hsi_by_cell(scores)
    show("OBSERVED HSI SCORES BY RACE x GENDER  (cell means; 2x2 model coefficients)",
         hsi, ["p_race", "p_gender", "p_interaction", "q_race", "q_gender", "q_interaction"])

    race = load_race()
    acc, direc = model_bias_by_cell(race)
    if direc is None:
        print("=" * 110)
        print("MODEL-SIDE ANALYSES SKIPPED")
        print("=" * 110)
        print("No pooled fold predictions found. Is the ACCOUNT share mounted?")
        print("The observed-score results above are complete and do not depend on it.\n")
    else:
        show("MODEL ACCURACY (r) BY RACE x GENDER CELL", acc, [])
        show("DIRECTIONAL BIAS WITH GENDER  (error ~ race * gender + true score)",
             direc, ["p_race", "p_gender", "p_interaction",
                     "q_race", "q_gender", "q_interaction"])

    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    frames = [("observed HSI", hsi)] + ([("directional bias", direc)] if direc is not None else [])
    for name, frame in frames:
        print(f"{name}:")
        for col in ["race", "gender", "interaction"]:
            print(f"    {col:12s} significant after BH on "
                  f"{int((frame[f'q_{col}'] < .05).sum())} of {len(frame)} outcomes")

    hsi.to_csv(f"{OUT}/intersectional_hsi_by_race_gender.csv", index=False)
    if direc is not None:
        acc.to_csv(f"{OUT}/intersectional_accuracy_by_cell.csv", index=False)
        direc.to_csv(f"{OUT}/intersectional_directional_bias.csv", index=False)
    print(f"\nSaved to {OUT}/intersectional_*.csv")


if __name__ == "__main__":
    main()
