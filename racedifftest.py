import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
import statsmodels.formula.api as smf

# ----------------------------------------------------------------------
# CONFIG -- edit these paths to match your machine
# ----------------------------------------------------------------------
BASE = "/path/to/project/Active/TD/SPAN"
RACE_JSON = os.path.join(BASE, "psm_combined.json")

# trait -> folder holding that trait's ffn_test_predictions_fold*.csv
# These are the PSM (propensity-matched) folders -- the paper's primary sample.
# The *mmc folders are the other matched sample and are intentionally NOT used here.
TRAIT_DIRS = {
    "Neuroticism":       os.path.join(BASE, "npsmc", "Modelnpsmc"),
    "Extraversion":      os.path.join(BASE, "epsmc", "Modelepsmc"),
    "Openness":          os.path.join(BASE, "opsmc", "Modelopsmc"),
    "Agreeableness":     os.path.join(BASE, "apsmc", "Modelapsmc"),
    "Conscientiousness": os.path.join(BASE, "cpsmc", "Modelcpsmc"),
}

RACE_COL = "PRACE"
RACE_MAP = {1: "White", 2: "Black"}
OUT_DIR = os.path.join(BASE, "race_difference_tests")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def r_to_z_test(r1, n1, r2, n2):
    """Two-independent-correlations z test. Returns (z, p_two_sided, cohens_q)."""
    z1, z2 = fisher_z(r1), fisher_z(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    q = abs(z1 - z2)  # Cohen's q
    return z, p, q

def load_pooled_trait(trait, folder):
    """Pool the 5 FFN test folds for a trait and merge race by PARTID."""
    files = sorted(glob.glob(os.path.join(folder, "ffn_test_predictions_fold*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"[{trait}] no ffn_test_predictions_fold*.csv in {folder}")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        need = {"PARTID", "true_label", "predicted_label"}
        if not need.issubset(d.columns):
            raise ValueError(f"[{trait}] {f} missing columns {need - set(d.columns)}")
        parts.append(d[["PARTID", "true_label", "predicted_label"]])
    pooled = pd.concat(parts, ignore_index=True)

    # sanity: each PARTID should appear exactly once across the 5 test folds (5-fold CV)
    dup = pooled["PARTID"].duplicated().sum()
    if dup > 0:
        print(f"  [WARN] {trait}: {dup} duplicate PARTIDs across test folds "
              f"(expected 0 for clean 5-fold CV). Check the fold files.")
    return pooled

# ----------------------------------------------------------------------
# Load race once
# ----------------------------------------------------------------------
race_df = pd.read_json(RACE_JSON, lines=True)
if "PARTID" not in race_df.columns or RACE_COL not in race_df.columns:
    raise ValueError(f"{RACE_JSON} must contain PARTID and {RACE_COL}")
race_df = race_df[["PARTID", RACE_COL]].drop_duplicates("PARTID")

# ----------------------------------------------------------------------
# Run per trait
# ----------------------------------------------------------------------
rtoz_rows = []
calib_rows = []

for trait, folder in TRAIT_DIRS.items():
    print(f"\n=== {trait} ===")
    pooled = load_pooled_trait(trait, folder)
    df = pooled.merge(race_df, on="PARTID", how="inner")
    df = df.dropna(subset=["true_label", "predicted_label", RACE_COL])
    df["race_name"] = df[RACE_COL].map(RACE_MAP)
    df = df[df["race_name"].isin(["White", "Black"])].copy()

    # ---- (1) Fisher r-to-z on predicted-true correlation, by race ----
    sub = {}
    for rn in ["White", "Black"]:
        s = df[df["race_name"] == rn]
        r, _ = pearsonr(s["true_label"], s["predicted_label"])
        sub[rn] = (r, len(s))
    (rW, nW), (rB, nB) = sub["White"], sub["Black"]
    z, p, q = r_to_z_test(rW, nW, rB, nB)
    print(f"  r_White={rW:.3f} (n={nW})  r_Black={rB:.3f} (n={nB})  "
          f"z={z:.2f}  p={p:.4f}  Cohen's q={q:.3f}")
    rtoz_rows.append({
        "Trait": trait, "r_White": round(rW, 3), "N_White": nW,
        "r_Black": round(rB, 3), "N_Black": nB,
        "z": round(z, 3), "p": round(p, 4), "Cohens_q": round(q, 3),
    })

    # ---- (2) Calibration interaction: true ~ predicted * race ----
    # race dummy: Black=1, White=0  (interaction = Black slope - White slope)
    df["race_black"] = (df["race_name"] == "Black").astype(int)
    model = smf.ols("true_label ~ predicted_label * race_black", data=df).fit()
    b_int = model.params["predicted_label:race_black"]
    p_int = model.pvalues["predicted_label:race_black"]
    slope_white = model.params["predicted_label"]
    slope_black = slope_white + b_int
    print(f"  calibration slope White={slope_white:.3f}  Black={slope_black:.3f}  "
          f"interaction b={b_int:.3f}  p={p_int:.4f}")
    calib_rows.append({
        "Trait": trait,
        "Slope_White": round(slope_white, 3),
        "Slope_Black": round(slope_black, 3),
        "Interaction_b": round(b_int, 3),
        "Interaction_p": round(p_int, 4),
    })

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
rtoz_df = pd.DataFrame(rtoz_rows)
calib_df = pd.DataFrame(calib_rows)

rtoz_path = os.path.join(OUT_DIR, "predicted_true_r_race_difference_fisher_z.csv")
calib_path = os.path.join(OUT_DIR, "calibration_slope_interaction_by_race.csv")
rtoz_df.to_csv(rtoz_path, index=False)
calib_df.to_csv(calib_path, index=False)

print("\n================ SUMMARY ================")
print("\nFisher r-to-z (predicted-true correlation, White vs Black):")
print(rtoz_df.to_string(index=False))
print("\nCalibration slope interaction (true ~ predicted * race):")
print(calib_df.to_string(index=False))
print(f"\nSaved:\n  {rtoz_path}\n  {calib_path}")