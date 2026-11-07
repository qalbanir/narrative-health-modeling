"""Multi-component bias analysis of the RoBERTa health-status predictions.

Implements the six components Tu's FFM paper specifies, on the pooled held-out
predictions from the five folds:

  1. Accuracy and discrimination  RMSE, R2 and Pearson r per group, with the
                                  group difference in r tested by Fisher r-to-z
  2. Mean bias                    mean(true - predicted) per group, tested
                                  against zero and between groups
  3. Distributional diagnostics   mean and variance of true and predicted scores
                                  per group; Levene's test on true-score variance
  4. Performance bias             difference in mean absolute error, a permutation
                                  test of that difference, and Cohen's d
  5. Directional bias             Error = b0 + b1(race) + b2(true score), with the
                                  partial R2 for race
  6. Calibration                  slope of true on predicted within each group,
                                  each tested against 1

SIGN CONVENTION: error is defined as true - predicted, matching Tu. A positive
mean bias therefore means UNDERprediction. Note this is the opposite of the
convention in rq2_results.py, which reports predicted - true.
"""

import glob
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import levene, pearsonr, t as t_dist, ttest_1samp, ttest_ind

from hsidata import LIWC_PATH, RACE_MAP, compare_correlations, ensure_out_dir
from rq2_results import OUTCOMES, RUNS_DIR

GROUPS = ["White", "Black"]
N_FOLDS = 5
N_PERMUTATIONS = 10000
RNG = np.random.default_rng(42)


def load_race():
    race = pd.read_csv(LIWC_PATH, low_memory=False, usecols=["PARTID", "PRACE"])
    race["race_name"] = race["PRACE"].map(RACE_MAP)
    return race[race["race_name"].isin(GROUPS)]


def pooled(stem, race):
    files = sorted(glob.glob(os.path.join(RUNS_DIR, stem, "Model*", "ffn_test_predictions_fold*.csv")))
    if len(files) < N_FOLDS:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.merge(race, on="PARTID", how="inner")
    df["error"] = df["true_label"] - df["predicted_label"]      # Tu's convention
    df["abs_error"] = df["error"].abs()
    df["black"] = (df["race_name"] == "Black").astype(int)
    return df


def r2_score(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def accuracy(df, outcome):
    row = {"HSI outcome": outcome}
    stats = {}
    for g in GROUPS:
        s = df[df["race_name"] == g]
        r, _ = pearsonr(s["true_label"], s["predicted_label"])
        stats[g] = (r, len(s))
        row.update({
            f"n_{g}": len(s),
            f"RMSE_{g}": round(np.sqrt((s["error"] ** 2).mean()), 3),
            f"R2_{g}": round(r2_score(s["true_label"], s["predicted_label"]), 3),
            f"r_{g}": round(r, 3),
        })
    (rW, nW), (rB, nB) = stats["White"], stats["Black"]
    z, p, q = compare_correlations(rB, nB, rW, nW)
    row.update({"z_r": round(z, 2), "p_r": p, "cohens_q": round(q, 3)})
    return row


def mean_bias(df, outcome):
    row = {"HSI outcome": outcome}
    for g in GROUPS:
        s = df[df["race_name"] == g]["error"]
        t, p = ttest_1samp(s, 0)
        row.update({f"bias_{g}": round(s.mean(), 3), f"t_{g}": round(t, 2), f"p_{g}": p})
    t, p = ttest_ind(df[df["race_name"] == "Black"]["error"],
                     df[df["race_name"] == "White"]["error"], equal_var=False)
    row.update({"t_diff": round(t, 2), "p_diff": p})
    return row


def distribution(df, outcome):
    row = {"HSI outcome": outcome}
    for g in GROUPS:
        s = df[df["race_name"] == g]
        row.update({
            f"true_M_{g}": round(s["true_label"].mean(), 3),
            f"true_Var_{g}": round(s["true_label"].var(ddof=1), 3),
            f"pred_M_{g}": round(s["predicted_label"].mean(), 3),
            f"pred_Var_{g}": round(s["predicted_label"].var(ddof=1), 3),
        })
    stat, p = levene(df[df["race_name"] == "White"]["true_label"],
                     df[df["race_name"] == "Black"]["true_label"])
    row.update({"levene_W": round(stat, 2), "levene_p": p,
                "var_ratio_B_over_W": round(row["true_Var_Black"] / row["true_Var_White"], 3)})
    return row


def performance_bias(df, outcome):
    """Difference in mean absolute error, permutation-tested."""
    w = df[df["race_name"] == "White"]["abs_error"].to_numpy()
    b = df[df["race_name"] == "Black"]["abs_error"].to_numpy()
    observed = b.mean() - w.mean()

    pooled_errors = np.concatenate([w, b])
    n_white = len(w)
    count = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = RNG.permutation(pooled_errors)
        delta = shuffled[n_white:].mean() - shuffled[:n_white].mean()
        if abs(delta) >= abs(observed):
            count += 1
    p_perm = (count + 1) / (N_PERMUTATIONS + 1)

    sd_pooled = np.sqrt(((len(w) - 1) * w.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(w) + len(b) - 2))
    d = observed / sd_pooled if sd_pooled > 0 else np.nan
    return {
        "HSI outcome": outcome,
        "MAE_White": round(w.mean(), 3),
        "MAE_Black": round(b.mean(), 3),
        "delta_MAE": round(observed, 3),
        "p_permutation": p_perm,
        "cohens_d": round(d, 3),
    }


def directional_bias(df, outcome):
    """Error ~ race + true score. b1 is directional bias net of true score."""
    full = smf.ols("error ~ black + true_label", data=df).fit()
    reduced = smf.ols("error ~ true_label", data=df).fit()
    partial_r2 = (reduced.ssr - full.ssr) / reduced.ssr if reduced.ssr > 0 else np.nan
    return {
        "HSI outcome": outcome,
        "b1_race": round(full.params["black"], 3),
        "SE": round(full.bse["black"], 3),
        "t": round(full.tvalues["black"], 2),
        "p": full.pvalues["black"],
        "partial_R2": round(partial_r2, 4),
        "b2_true": round(full.params["true_label"], 3),
    }


def calibration(df, outcome):
    """Slope of true on predicted within each group, tested against 1."""
    row = {"HSI outcome": outcome}
    for g in GROUPS:
        s = df[df["race_name"] == g]
        fit = smf.ols("true_label ~ predicted_label", data=s).fit()
        slope, se = fit.params["predicted_label"], fit.bse["predicted_label"]
        t_stat = (slope - 1) / se
        p = 2 * (1 - t_dist.cdf(abs(t_stat), fit.df_resid))
        row.update({f"slope_{g}": round(slope, 3), f"t_vs1_{g}": round(t_stat, 2),
                    f"p_vs1_{g}": p})
    return row


def fmt_p(v):
    return "< .001" if v < 0.001 else f"{v:.3f}".lstrip("0")


def show(title, frame, p_cols):
    print(f"\n--- {title} ---")
    out = frame.copy()
    for c in p_cols:
        out[c] = out[c].map(fmt_p)
    print(out.to_string(index=False))


def main():
    out_dir = ensure_out_dir()
    if not os.path.isdir(RUNS_DIR):
        raise SystemExit(f"Cannot reach {RUNS_DIR}\nIs the ACCOUNT share mounted?")

    race = load_race()
    tables = {k: [] for k in ["accuracy", "mean_bias", "distribution",
                              "performance", "directional", "calibration"]}
    pending = []

    for stem, outcome in OUTCOMES.items():
        df = pooled(stem, race)
        if df is None:
            pending.append(outcome)
            continue
        tables["accuracy"].append(accuracy(df, outcome))
        tables["mean_bias"].append(mean_bias(df, outcome))
        tables["distribution"].append(distribution(df, outcome))
        tables["performance"].append(performance_bias(df, outcome))
        tables["directional"].append(directional_bias(df, outcome))
        tables["calibration"].append(calibration(df, outcome))

    if not tables["accuracy"]:
        raise SystemExit("No completed runs found.")

    frames = {k: pd.DataFrame(v) for k, v in tables.items()}
    pd.set_option("display.width", 260, "display.max_columns", 40)

    print("=" * 118)
    print("BIAS ANALYSES -- pooled held-out predictions, combined-sample models")
    print("Error is defined as true - predicted; positive means UNDERprediction.")
    print("=" * 118)

    show("1. ACCURACY AND DISCRIMINATION", frames["accuracy"], ["p_r"])
    show("2. MEAN BIAS", frames["mean_bias"], ["p_White", "p_Black", "p_diff"])
    show("3. DISTRIBUTIONAL DIAGNOSTICS", frames["distribution"], ["levene_p"])
    show("4. PERFORMANCE BIAS (absolute error)", frames["performance"], ["p_permutation"])
    show("5. DIRECTIONAL BIAS (net of true score)", frames["directional"], ["p"])
    show("6. CALIBRATION (slope vs 1)", frames["calibration"], ["p_vs1_White", "p_vs1_Black"])

    print("\n" + "=" * 118)
    print("SUMMARY (p < .05)")
    print("=" * 118)
    n = len(frames["accuracy"])
    checks = [
        ("accuracy differs by race", (frames["accuracy"]["p_r"] < .05).sum()),
        ("mean bias differs by race", (frames["mean_bias"]["p_diff"] < .05).sum()),
        ("true-score variance differs (Levene)", (frames["distribution"]["levene_p"] < .05).sum()),
        ("absolute error differs (permutation)", (frames["performance"]["p_permutation"] < .05).sum()),
        ("directional bias net of true score", (frames["directional"]["p"] < .05).sum()),
        ("calibration slope differs from 1, White", (frames["calibration"]["p_vs1_White"] < .05).sum()),
        ("calibration slope differs from 1, Black", (frames["calibration"]["p_vs1_Black"] < .05).sum()),
    ]
    for label, count in checks:
        print(f"  {label:44s} {count} of {n}")

    for name, frame in frames.items():
        frame.to_csv(f"{out_dir}/bias_{name}.csv", index=False)
    print(f"\nSaved six tables to {out_dir}/bias_*.csv")
    if pending:
        print(f"Incomplete ({len(pending)}): {', '.join(pending)}")


if __name__ == "__main__":
    main()
