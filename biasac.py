# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import probplot, ks_2samp, levene, pearsonr, t as tdist, ttest_1samp
import matplotlib.pyplot as plt

############################################################
# CONFIG
############################################################

# Directory that contains your ffn_predictions_fold*.csv
try:
    output_dir
except NameError:
    output_dir = "/path/to/project/Active/TD/SPAN/ac/ModelAc/"

print("Using output_dir:", output_dir)

# Path to original data with PARTID + PRACE
# Change this if your file lives somewhere else
race_data_path = "/path/to/project/Active/TD/SPAN/combined_data_prace.json"

# Glob pattern for prediction files (FFN)
cv_pred_pattern = os.path.join(output_dir, "ffn_predictions_fold*.csv")
cv_pred_files = sorted(glob(cv_pred_pattern))

if not cv_pred_files:
    raise FileNotFoundError("No ffn_predictions_fold*.csv files found in output_dir.")

print("Found CV prediction files:")
for f in cv_pred_files:
    print("  ", f)

############################################################
# LOAD RACE INFO (PARTID + PRACE)
############################################################
print("\nLoading race info from:", race_data_path)
race_df = pd.read_json(race_data_path, lines=True)

if "PARTID" not in race_df.columns or "PRACE" not in race_df.columns:
    raise ValueError("Race data file must contain PARTID and PRACE columns.")

race_df = race_df[["PARTID", "PRACE"]].drop_duplicates(subset="PARTID")

print("Race data rows:", race_df.shape[0])
print("PRACE value counts in race data:")
print(race_df["PRACE"].value_counts(dropna=False))

############################################################
# LOAD AND COMBINE ALL FOLDS, THEN MERGE RACE
############################################################
all_rows = []
for fpath in cv_pred_files:
    fname = os.path.splitext(os.path.basename(fpath))[0]  # e.g., "ffn_predictions_fold1"
    try:
        fold_id = int(fname.split("fold")[-1])
    except ValueError:
        raise ValueError("Could not parse fold id from file name: %s" % fname)

    df = pd.read_csv(fpath)
    df["fold"] = fold_id
    all_rows.append(df)

all_df = pd.concat(all_rows, ignore_index=True)

# Merge in PRACE by PARTID
all_df = all_df.merge(race_df, on="PARTID", how="left")

missing_race = all_df["PRACE"].isna().sum()
if missing_race > 0:
    print("\nWarning: %d prediction rows have no PRACE after merge; they will be dropped." % missing_race)
    all_df = all_df[~all_df["PRACE"].isna()].copy()

# Keep only White (1) and Black (2)
all_df = all_df[all_df["PRACE"].isin([1, 2])].copy()
all_df["race_name"] = all_df["PRACE"].map({1: "White", 2: "Black"})
all_df["race_black"] = (all_df["PRACE"] == 2).astype(int)

print("\nTotal pooled rows after merge and filtering to White/Black:", len(all_df))
print(all_df["race_name"].value_counts())

############################################################
# HELPER: PERFORMANCE METRICS
############################################################
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    if len(y_true) > 1:
        r2 = r2_score(y_true, y_pred)
        r, p = pearsonr(y_true, y_pred)
    else:
        r2, r, p = np.nan, np.nan, np.nan
    var_pred = float(np.var(y_pred))
    calib = float(np.mean(y_pred - y_true))   # mean(pred - true)
    return {
        "N": len(y_true),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Pearson_r": r,
        "Pearson_p": p,
        "Var_pred": var_pred,
        "Calibration_Error": calib
    }

############################################################
# HELPER: MANUAL OLS (NO STATSMODELS)
############################################################
def ols_manual(X, y):
    """
    Return beta, se_beta, t_stats, p_vals, R2
    """
    XtX = X.T.dot(X)
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T.dot(y)
    beta = XtX_inv.dot(Xty)

    y_pred = X.dot(beta)
    residuals = y - y_pred
    n = len(y)
    k = X.shape[1]

    sse = float(np.dot(residuals, residuals))
    sst = float(np.dot(y - np.mean(y), y - np.mean(y)))

    sigma2 = sse / (n - k) if (n - k) > 0 else np.nan
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))

    t_stats = beta / se_beta
    df = max(n - k, 1)
    p_vals = 2 * (1 - tdist.cdf(np.abs(t_stats), df))

    R2 = 1 - sse / sst if sst > 0 else np.nan

    return beta, se_beta, t_stats, p_vals, R2

############################################################
# HELPER: DELTA_MAE (Black - White)
############################################################
def delta_mae(df):
    black = df[df["race_black"] == 1]
    white = df[df["race_black"] == 0]
    if len(black) == 0 or len(white) == 0:
        return np.nan
    mae_b = mean_absolute_error(black["true_label"], black["predicted_label"])
    mae_w = mean_absolute_error(white["true_label"], white["predicted_label"])
    return mae_b - mae_w

############################################################
# STEP 1 - METRICS BY RACE (PER-FOLD + POOLED)
############################################################
print("\n" + "="*70)
print("STEP 1: METRICS BY RACE")
print("="*70)

step1_rows = []
for fold_id, df_fold in all_df.groupby("fold"):
    for race_val, race_name in [(1, "White"), (2, "Black")]:
        sub = df_fold[df_fold["PRACE"] == race_val]
        if len(sub) == 0:
            continue
        m = compute_metrics(sub["true_label"].values,
                            sub["predicted_label"].values)
        row = {"fold": fold_id, "race": race_name}
        row.update(m)
        step1_rows.append(row)

step1_df = pd.DataFrame(step1_rows).sort_values(["fold", "race"])
step1_path = os.path.join(output_dir, "bias_step1_metrics_by_race_per_fold.csv")
step1_df.to_csv(step1_path, index=False)
print("Saved:", step1_path)

# Pooled
pooled_rows = []
for race_val, race_name in [(1, "White"), (2, "Black")]:
    sub = all_df[all_df["PRACE"] == race_val]
    m = compute_metrics(sub["true_label"].values,
                        sub["predicted_label"].values)
    row = {"race": race_name}
    row.update(m)
    pooled_rows.append(row)

pooled_step1_df = pd.DataFrame(pooled_rows)
pooled_path = os.path.join(output_dir, "bias_step1_metrics_by_race_pooled.csv")
pooled_step1_df.to_csv(pooled_path, index=False)
print("Saved pooled:", pooled_path)

############################################################
# STEP 2 - ERROR REGRESSION (NO STATSMODELS)
############################################################
print("\n" + "="*70)
print("STEP 2: ERROR REGRESSION (no statsmodels)")
print("="*70)

all_df["error"] = all_df["true_label"] - all_df["predicted_label"]

step2_rows = []
for fold_id, df_fold in all_df.groupby("fold"):
    y = df_fold["error"].values
    X = np.column_stack([
        np.ones(len(df_fold)),         # intercept
        df_fold["race_black"].values,  # 1 = Black
        df_fold["true_label"].values   # trait level
    ])

    beta, se, tstat, pvals, R2 = ols_manual(X, y)

    step2_rows.append({
        "fold": fold_id,
        "b_intercept": beta[0],
        "b_race_black": beta[1],
        "b_true_label": beta[2],
        "p_race_black": pvals[1],
        "R2_model": R2,
        "N": len(df_fold)
    })

step2_df = pd.DataFrame(step2_rows).sort_values("fold")
step2_path = os.path.join(output_dir, "bias_step2_error_regression_per_fold.csv")
step2_df.to_csv(step2_path, index=False)
print("Saved:", step2_path)

# Pooled
y_all = all_df["error"].values
X_all = np.column_stack([
    np.ones(len(all_df)),
    all_df["race_black"].values,
    all_df["true_label"].values
])

beta_all, se_all, t_all, p_all, R2_all = ols_manual(X_all, y_all)

pooled_step2_df = pd.DataFrame([{
    "b_intercept": beta_all[0],
    "b_race_black": beta_all[1],
    "b_true_label": beta_all[2],
    "p_race_black": p_all[1],
    "R2_model": R2_all,
    "N": len(all_df)
}])

pooled_step2_path = os.path.join(output_dir, "bias_step2_error_regression_pooled.csv")
pooled_step2_df.to_csv(pooled_step2_path, index=False)
print("Saved pooled:", pooled_step2_path)

############################################################
# STEP 3 - PERMUTATION TEST ON DELTA MAE
############################################################
print("\n" + "="*70)
print("STEP 3: PERMUTATION TEST ON DELTA MAE (Black - White)")
print("="*70)

n_perm = 5000
np.random.seed(42)

step3_rows = []
for fold_id, df_fold in all_df.groupby("fold"):
    observed = delta_mae(df_fold)
    if np.isnan(observed):
        continue

    perm_deltas = []
    for _ in range(n_perm):
        shuffled = df_fold.copy()
        shuffled["race_black"] = np.random.permutation(shuffled["race_black"].values)
        perm_deltas.append(delta_mae(shuffled))

    perm_deltas = np.array(perm_deltas)
    p_val = np.mean(np.abs(perm_deltas) >= np.abs(observed))

    step3_rows.append({
        "fold": fold_id,
        "observed_delta_MAE": observed,
        "p_value": p_val,
        "perm_mean_delta": float(np.mean(perm_deltas)),
        "perm_std_delta": float(np.std(perm_deltas)),
        "N": len(df_fold)
    })

        # Optional: save histogram for each fold
    plt.figure()
    plt.hist(perm_deltas, bins=40)
    plt.axvline(observed, linestyle="--", color="red")
    plt.title("Permutation Delta MAE (Black - White), Fold %d" % fold_id)
    plt.xlabel("Delta MAE")
    plt.ylabel("Count")
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "perm_delta_mae_fold%d.png" % fold_id)
    plt.savefig(hist_path)
    plt.close()


step3_df = pd.DataFrame(step3_rows).sort_values("fold")
step3_path = os.path.join(output_dir, "bias_step3_permutation_per_fold.csv")
step3_df.to_csv(step3_path, index=False)
print("Saved:", step3_path)

# Pooled permutation
observed_pooled = delta_mae(all_df)
perm_deltas_pooled = []
for _ in range(n_perm):
    shuffled_all = all_df.copy()
    shuffled_all["race_black"] = np.random.permutation(shuffled_all["race_black"].values)
    perm_deltas_pooled.append(delta_mae(shuffled_all))

perm_deltas_pooled = np.array(perm_deltas_pooled)
p_val_pooled = np.mean(np.abs(perm_deltas_pooled) >= np.abs(observed_pooled))

pooled_step3_df = pd.DataFrame([{
    "observed_delta_MAE": observed_pooled,
    "p_value": p_val_pooled,
    "perm_mean_delta": float(np.mean(perm_deltas_pooled)),
    "perm_std_delta": float(np.std(perm_deltas_pooled)),
    "N": len(all_df)
}])

pooled_step3_path = os.path.join(output_dir, "bias_step3_permutation_delta_mae_pooled.csv")
pooled_step3_df.to_csv(pooled_step3_path, index=False)

plt.figure()
plt.hist(perm_deltas_pooled, bins=40)
plt.axvline(observed_pooled, linestyle="--", color="red")
plt.title("Permutation Delta MAE (Black - White), Pooled")
plt.xlabel("Delta MAE")
plt.ylabel("Count")
plt.tight_layout()
pooled_hist_path = os.path.join(output_dir, "perm_delta_mae_pooled.png")
plt.savefig(pooled_hist_path)
plt.close()

print("\nPooled permutation results saved to:", pooled_step3_path)
print(pooled_step3_df)

print("\n===============================")
print("BIAS ANALYSIS COMPLETED")
print("===============================")

############################################################
# EXTRA ANALYSIS A: COHEN'S D (BLACK - WHITE)
############################################################

def cohens_d(group1, group2):
    """Cohen's d (group1 - group2) using pooled SD."""
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1 = np.std(group1, ddof=1)
    s2 = np.std(group2, ddof=1)
    # pooled SD
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / s_pooled

# Make sure we have error column (should already exist from Step 2)
if "error" not in all_df.columns:
    all_df["error"] = all_df["true_label"] - all_df["predicted_label"]

all_df["abs_error"] = np.abs(all_df["error"])

# Per-fold Cohen's d: Black - White
cohen_rows = []
for fold_id, df_fold in all_df.groupby("fold"):
    black = df_fold[df_fold["race_black"] == 1]
    white = df_fold[df_fold["race_black"] == 0]

    if len(black) < 2 or len(white) < 2:
        # Not enough data in this fold
        continue

    d_abs_error = cohens_d(black["abs_error"], white["abs_error"])
    d_pred = cohens_d(black["predicted_label"], white["predicted_label"])

    cohen_rows.append({
        "fold": fold_id,
        "d_abs_error_Black_minus_White": d_abs_error,
        "d_pred_Black_minus_White": d_pred,
        "N_black": len(black),
        "N_white": len(white)
    })

cohen_df = pd.DataFrame(cohen_rows).sort_values("fold")
cohen_path = os.path.join(output_dir, "bias_step_effectsizes_cohens_d_per_fold.csv")
cohen_df.to_csv(cohen_path, index=False)
print("Saved Cohen's d per fold:", cohen_path)

# Pooled Cohen's d using all pooled out-of-fold predictions
black_all = all_df[all_df["race_black"] == 1]
white_all = all_df[all_df["race_black"] == 0]

d_abs_error_pooled = cohens_d(black_all["abs_error"], white_all["abs_error"])
d_pred_pooled = cohens_d(black_all["predicted_label"], white_all["predicted_label"])

cohen_pooled_df = pd.DataFrame([{
    "d_abs_error_Black_minus_White": d_abs_error_pooled,
    "d_pred_Black_minus_White": d_pred_pooled,
    "N_black": len(black_all),
    "N_white": len(white_all)
}])

cohen_pooled_path = os.path.join(output_dir, "bias_step_effectsizes_cohens_d_pooled.csv")
cohen_pooled_df.to_csv(cohen_pooled_path, index=False)
print("Saved pooled Cohen's d:", cohen_pooled_path)


############################################################
# EXTRA ANALYSIS B: FOLD-LEVEL SUMMARY (MEAN ± SD)
############################################################

# 1) Summary of metrics per race across folds
metrics_cols = ["MAE", "MSE", "RMSE", "R2", "Pearson_r",
                "Var_pred", "Calibration_Error"]

summary_rows = []
for race_name in ["White", "Black"]:
    sub = step1_df[step1_df["race"] == race_name]
    if sub.empty:
        continue
    for m in metrics_cols:
        mean_val = sub[m].mean()
        sd_val = sub[m].std(ddof=1)
        summary_rows.append({
            "race": race_name,
            "metric": m,
            "mean": mean_val,
            "sd": sd_val,
            "n_folds": sub["fold"].nunique()
        })

fold_summary_df = pd.DataFrame(summary_rows)
fold_summary_path = os.path.join(output_dir, "bias_step_fold_summary_metrics_by_race.csv")
fold_summary_df.to_csv(fold_summary_path, index=False)
print("Saved fold-level summary of metrics:", fold_summary_path)

# 2) Summary of Cohen's d across folds
effects_summary_rows = []
if not cohen_df.empty:
    for metric_name in ["d_abs_error_Black_minus_White",
                        "d_pred_Black_minus_White"]:
        mean_val = cohen_df[metric_name].mean()
        sd_val = cohen_df[metric_name].std(ddof=1)
        effects_summary_rows.append({
            "contrast": "Black_minus_White",
            "effect_metric": metric_name,
            "mean_d": mean_val,
            "sd_d": sd_val,
            "n_folds": cohen_df["fold"].nunique()
        })

effects_summary_df = pd.DataFrame(effects_summary_rows)
effects_summary_path = os.path.join(output_dir, "bias_step_fold_summary_effectsizes.csv")
effects_summary_df.to_csv(effects_summary_path, index=False)
print("Saved fold-level summary of effect sizes:", effects_summary_path)

############################################################
# EXTRA ANALYSIS C: ONE-SAMPLE T-TEST ON ΔMAE AND ΔR² ACROSS FOLDS
############################################################

delta_mae_folds = []
delta_r2_folds = []

for fold_id in sorted(step1_df["fold"].unique()):
    sub = step1_df[step1_df["fold"] == fold_id]
    white_row = sub[sub["race"] == "White"]
    black_row = sub[sub["race"] == "Black"]

    if white_row.empty or black_row.empty:
        continue

    mae_w = white_row["MAE"].values[0]
    mae_b = black_row["MAE"].values[0]
    r2_w = white_row["R2"].values[0]
    r2_b = black_row["R2"].values[0]

    # Black - White differences per fold
    delta_mae_folds.append(mae_b - mae_w)
    delta_r2_folds.append(r2_b - r2_w)

ttest_rows = []

# MAE
if len(delta_mae_folds) > 1:
    t_mae, p_mae = ttest_1samp(delta_mae_folds, 0.0)
    ttest_rows.append({
        "metric": "MAE (Black - White)",
        "N_folds": len(delta_mae_folds),
        "mean_delta": float(np.mean(delta_mae_folds)),
        "sd_delta": float(np.std(delta_mae_folds, ddof=1)),
        "t_stat": float(t_mae),
        "p_value": float(p_mae)
    })

# R2
if len(delta_r2_folds) > 1:
    t_r2, p_r2 = ttest_1samp(delta_r2_folds, 0.0)
    ttest_rows.append({
        "metric": "R2 (Black - White)",
        "N_folds": len(delta_r2_folds),
        "mean_delta": float(np.mean(delta_r2_folds)),
        "sd_delta": float(np.std(delta_r2_folds, ddof=1)),
        "t_stat": float(t_r2),
        "p_value": float(p_r2)
    })

ttest_df = pd.DataFrame(ttest_rows)
ttest_path = os.path.join(output_dir, "bias_step_foldwise_ttests_deltas.csv")
ttest_df.to_csv(ttest_path, index=False)
print("Saved fold-wise t-tests on deltas:", ttest_path)

############################################################
# EXTRA ANALYSIS D: ERROR DISTRIBUTION PARITY
############################################################

print("\n=== ERROR DISTRIBUTION PARITY ANALYSIS ===")

# Ensure we have errors
if "error" not in all_df.columns:
    all_df["error"] = all_df["true_label"] - all_df["predicted_label"]

# Separate errors
err_black = all_df[all_df["race_black"] == 1]["error"].values
err_white = all_df[all_df["race_black"] == 0]["error"].values

# --- KS test (distribution equality)
ks_stat, ks_p = ks_2samp(err_black, err_white)

# --- Levene test (variance parity)
lev_stat, lev_p = levene(err_black, err_white)

print(f"Kolmogorov–Smirnov test: statistic={ks_stat:.4f}, p={ks_p:.4f}")
print(f"Levene variance test:    statistic={lev_stat:.4f}, p={lev_p:.4f}")

# --- Save histogram plot
plt.figure(figsize=(8,5))
plt.hist(err_white, bins=30, alpha=0.6, label="White")
plt.hist(err_black, bins=30, alpha=0.6, label="Black")
plt.title("Error Distribution: Black vs White")
plt.xlabel("Error (true - predicted)")
plt.ylabel("Count")
plt.legend()
hist_path = os.path.join(output_dir, "error_hist_black_vs_white.png")
plt.savefig(hist_path, dpi=150)
plt.close()
print("Saved histogram:", hist_path)

# --- QQ Plot
plt.figure(figsize=(8,5))
probplot(err_white, dist="norm", plot=plt)
qq_white_path = os.path.join(output_dir, "qqplot_white_errors.png")
plt.title("QQ Plot – White Errors")
plt.savefig(qq_white_path, dpi=150)
plt.close()

plt.figure(figsize=(8,5))
probplot(err_black, dist="norm", plot=plt)
qq_black_path = os.path.join(output_dir, "qqplot_black_errors.png")
plt.title("QQ Plot – Black Errors")
plt.savefig(qq_black_path, dpi=150)
plt.close()

print("Saved QQ plots:")
print(" -", qq_white_path)
print(" -", qq_black_path)
