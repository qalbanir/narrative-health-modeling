"""RQ2: does the fine-tuned model perform differently by race?

Reads whatever roberta_runs/ contains and, for every outcome with a complete set
of five folds, pools the held-out FFN predictions and tests the three things RQ2
asks about:

  accuracy     Pearson r within each race, compared with Fisher r-to-z
  mean bias    mean(predicted - true) per race, tested against zero and
               against each other
  calibration  true ~ predicted * race; the interaction is the slope difference

Outcomes still running are reported as incomplete and skipped, so this can be run
at any time.
"""

import glob
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, ttest_1samp, ttest_ind

from hsidata import LIWC_PATH, RACE_MAP, compare_correlations, ensure_out_dir

RUNS_DIR = "/path/to/project/roberta_runs"

# folder stem -> HSI outcome, matching make_hsi_scripts.py
OUTCOMES = {
    "pfpsmc": "Physical functioning",
    "rppsmc": "Role limitations due to physical health",
    "repsmc": "Role limitations due to emotional problems",
    "efpsmc": "Energy/fatigue",
    "ewpsmc": "Emotional well-being",
    "sfpsmc": "Social Functioning",
    "pnpsmc": "Pain",
    "ghpsmc": "General health",
    "phcpsmc": "Physical Health Composite",
    "mhcpsmc": "Mental Health Composite",
}

GROUPS = ["White", "Black"]
N_FOLDS = 5


def load_race():
    race = pd.read_csv(LIWC_PATH, low_memory=False, usecols=["PARTID", "PRACE"])
    race["race_name"] = race["PRACE"].map(RACE_MAP)
    return race[race["race_name"].isin(GROUPS)]


def pooled_predictions(stem):
    """Concatenate the five held-out test folds, or None if incomplete."""
    pattern = os.path.join(RUNS_DIR, stem, "Model*", "ffn_test_predictions_fold*.csv")
    files = sorted(glob.glob(pattern))
    if len(files) < N_FOLDS:
        return None, len(files)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df, len(files)


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def analyse(stem, outcome, race):
    preds, n_folds = pooled_predictions(stem)
    if preds is None:
        return None, n_folds

    df = preds.merge(race, on="PARTID", how="inner")
    df["error"] = df["predicted_label"] - df["true_label"]

    row = {"HSI outcome": outcome, "N": len(df), "dup_PARTID": int(df["PARTID"].duplicated().sum())}

    stats = {}
    for group in GROUPS:
        s = df[df["race_name"] == group]
        r, _ = pearsonr(s["true_label"], s["predicted_label"])
        t_bias, p_bias = ttest_1samp(s["error"], 0)
        stats[group] = (r, len(s))
        row.update({
            f"n_{group}": len(s),
            f"r_{group}": round(r, 3),
            f"R2_{group}": round(r2(s["true_label"], s["predicted_label"]), 3),
            f"MAE_{group}": round(s["error"].abs().mean(), 3),
            f"bias_{group}": round(s["error"].mean(), 3),
            f"bias_p_{group}": p_bias,
        })

    (rW, nW), (rB, nB) = stats["White"], stats["Black"]
    z, p_acc, q = compare_correlations(rB, nB, rW, nW)
    row.update({"z_accuracy": round(z, 2), "p_accuracy": p_acc, "cohens_q": round(q, 3)})

    t_diff, p_diff = ttest_ind(
        df[df["race_name"] == "White"]["error"],
        df[df["race_name"] == "Black"]["error"],
        equal_var=False,
    )
    row.update({"t_bias_diff": round(t_diff, 2), "p_bias_diff": p_diff})

    df["black"] = (df["race_name"] == "Black").astype(int)
    fit = smf.ols("true_label ~ predicted_label * black", data=df).fit()
    b = fit.params["predicted_label:black"]
    slope_w = fit.params["predicted_label"]
    row.update({
        "slope_White": round(slope_w, 3),
        "slope_Black": round(slope_w + b, 3),
        "calib_b": round(b, 3),
        "p_calibration": fit.pvalues["predicted_label:black"],
    })
    return row, n_folds


def fmt_p(v):
    return "< .001" if v < 0.001 else f"{v:.3f}".lstrip("0")


def main():
    out_dir = ensure_out_dir()
    if not os.path.isdir(RUNS_DIR):
        raise SystemExit(f"Cannot reach {RUNS_DIR}\nIs the ACCOUNT share mounted?")

    race = load_race()
    rows, pending = [], []

    for stem, outcome in OUTCOMES.items():
        row, n_folds = analyse(stem, outcome, race)
        if row is None:
            pending.append((outcome, n_folds))
        else:
            rows.append(row)

    if not rows:
        print("No outcome has all five folds yet.")
        for outcome, n in pending:
            print(f"  {outcome}: {n}/{N_FOLDS} folds")
        return

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 40)

    print("=" * 108)
    print("RQ2. Model performance by race, pooled across 5 held-out folds")
    print("=" * 108)

    acc = results[["HSI outcome", "n_White", "r_White", "R2_White", "MAE_White",
                   "n_Black", "r_Black", "R2_Black", "MAE_Black",
                   "z_accuracy", "p_accuracy"]].copy()
    acc["p_accuracy"] = acc["p_accuracy"].map(fmt_p)
    print("\n--- 1. ACCURACY ---")
    print(acc.to_string(index=False))

    bias = results[["HSI outcome", "bias_White", "bias_p_White",
                    "bias_Black", "bias_p_Black", "t_bias_diff", "p_bias_diff"]].copy()
    for c in ["bias_p_White", "bias_p_Black", "p_bias_diff"]:
        bias[c] = bias[c].map(fmt_p)
    print("\n--- 2. MEAN-LEVEL BIAS (predicted - true; positive = overprediction) ---")
    print(bias.to_string(index=False))

    calib = results[["HSI outcome", "slope_White", "slope_Black",
                     "calib_b", "p_calibration"]].copy()
    calib["p_calibration"] = calib["p_calibration"].map(fmt_p)
    print("\n--- 3. CALIBRATION (true ~ predicted * race) ---")
    print(calib.to_string(index=False))

    print("\n" + "=" * 108)
    print("SUMMARY")
    print("=" * 108)
    n = len(results)
    print(f"Outcomes analysed: {n} of {len(OUTCOMES)}")
    print(f"  accuracy differs by race (p < .05):    {(results['p_accuracy'] < .05).sum()} of {n}")
    print(f"  mean bias differs by race (p < .05):   {(results['p_bias_diff'] < .05).sum()} of {n}")
    print(f"  calibration differs by race (p < .05): {(results['p_calibration'] < .05).sum()} of {n}")
    if results["dup_PARTID"].sum():
        print(f"  WARNING: duplicate PARTIDs found -- check the fold files")

    if pending:
        print(f"\nStill running ({len(pending)}):")
        for outcome, n_folds in pending:
            print(f"  {outcome}: {n_folds}/{N_FOLDS} folds")

    path = f"{out_dir}/rq2_model_performance_by_race.csv"
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
