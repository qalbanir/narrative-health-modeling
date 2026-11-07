"""Shared data loading for the Health Status Inventory (RAND SF-36) analyses.

Merges the propensity-matched LIWC sample with the scored HSI subscales and
participant baseline demographics, keyed on PARTID.
"""

import os
import numpy as np
import pandas as pd

DOWNLOADS = os.path.expanduser("~/Downloads")

LIWC_PATH = os.path.join(DOWNLOADS, "liwcpsmfile (1).csv")
HSI_PATH = os.path.join(DOWNLOADS, "PHSI .xlsx")
HSI_ITEM_PATH = os.path.join(DOWNLOADS, "PHSI (rescored).xlsx")
DEMO_PATH = os.path.join(DOWNLOADS, "PDEMOtopic.csv")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

RACE_COL = "PRACE"
RACE_MAP = {1: "White", 2: "Black"}

LIWC_VARS = [
    "health",
    "illness",
    "wellness",
    "mental",
    "Physical",
    "feeling",
    "fatigue",
    "emo_neg",
    "emo_anx",
    "emo_sad",
    "emo_pos",
    "tone_pos",
]

HSI_SUBSCALES = [
    "Physical functioning",
    "Role limitations due to physical health",
    "Role limitations due to emotional problems",
    "Energy/fatigue",
    "Emotional well-being",
    "Social Functioning",
    "Pain",
    "General health",
]

# Standard RAND-36 grouping of the eight subscales into physical and mental summaries.
HSI_COMPOSITES = {
    "Physical Health Composite": [
        "Physical functioning",
        "Role limitations due to physical health",
        "Pain",
        "General health",
    ],
    "Mental Health Composite": [
        "Role limitations due to emotional problems",
        "Energy/fatigue",
        "Emotional well-being",
        "Social Functioning",
    ],
}

HSI_OUTCOMES = HSI_SUBSCALES + list(HSI_COMPOSITES)

# Standard RAND-36 item-to-subscale assignment; verified to reproduce the scored
# subscale file exactly (max abs difference 0.0000 on all eight scales).
RAND_ITEMS = {
    "Physical functioning": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Role limitations due to physical health": [13, 14, 15, 16],
    "Role limitations due to emotional problems": [17, 18, 19],
    "Energy/fatigue": [23, 27, 29, 31],
    "Emotional well-being": [24, 25, 26, 28, 30],
    "Social Functioning": [20, 32],
    "Pain": [21, 22],
    "General health": [1, 33, 34, 35, 36],
}

for _name, _members in HSI_COMPOSITES.items():
    RAND_ITEMS[_name] = sorted(i for m in _members for i in RAND_ITEMS[m])

DEMO_VARS = {
    "PAGE": "Age",
    "PGENDER": "Gender",
    "PMariStat": "Marital status",
    "PEdu": "Education",
    "PHAnnIncome": "Annual household income",
}

DEMO_LABELS = {
    "PGENDER": {1: "Male", 2: "Female"},
    "PMariStat": {
        1: "Single (never married)",
        2: "Married",
        3: "Living with partner, not married",
        4: "Other serious relationship",
        5: "Separated",
        6: "Divorced",
        7: "Widowed",
    },
    "PEdu": {
        1: "Less than high school",
        2: "High school (or GED)",
        3: "Some college",
        4: "Vocational school",
        5: "2-year college degree",
        6: "4-year college degree",
        7: "Master's degree",
        8: "Doctoral degree",
        9: "Professional degree (MD, JD)",
    },
    "PHAnnIncome": {
        1: "Under $20,000",
        2: "$20,000 - $39,999",
        3: "$40,000 - $59,999",
        4: "$60,000 - $79,999",
        5: "$80,000 - $99,999",
        6: "$100,000 - $119,999",
        7: "$120,000 - $139,999",
        8: "$140,000 or more",
    },
}

CATEGORICAL_DEMOS = ["PGENDER", "PMariStat", "PEdu", "PHAnnIncome"]
CONTINUOUS_DEMOS = ["PAGE"]


def load_data(verbose=True):
    """Return the merged analysis frame restricted to Black and White participants."""
    liwc = pd.read_csv(LIWC_PATH, low_memory=False)
    hsi = pd.read_excel(HSI_PATH, usecols="A:I")
    demo = pd.read_csv(DEMO_PATH, low_memory=False)

    liwc = liwc[["PARTID", RACE_COL] + LIWC_VARS].copy()
    for var in LIWC_VARS:
        liwc[var] = pd.to_numeric(liwc[var], errors="coerce")

    hsi["PARTID"] = pd.to_numeric(hsi["PARTID"], errors="coerce")
    hsi = hsi.dropna(subset=["PARTID"]).drop_duplicates("PARTID")
    for var in HSI_SUBSCALES:
        hsi[var] = pd.to_numeric(hsi[var], errors="coerce")

    demo = demo[["PARTID"] + list(DEMO_VARS)].drop_duplicates("PARTID").copy()
    for var in DEMO_VARS:
        demo[var] = pd.to_numeric(demo[var], errors="coerce")

    df = liwc.merge(hsi, on="PARTID", how="left").merge(demo, on="PARTID", how="left")

    for name, members in HSI_COMPOSITES.items():
        df[name] = df[members].mean(axis=1, skipna=False)

    df["race_name"] = df[RACE_COL].map(RACE_MAP)
    df = df[df["race_name"].isin(["White", "Black"])].copy()

    if verbose:
        print(f"Merged analysis sample: {len(df)} participants")
        print(df["race_name"].value_counts().to_string())
        n_hsi = df[HSI_SUBSCALES].notna().all(axis=1).sum()
        n_demo = df["PAGE"].notna().sum()
        print(f"Complete on all 8 HSI subscales: {n_hsi}")
        print(f"Has baseline demographics: {n_demo}")

    return df


def load_items(df=None):
    """Return per-participant RAND-36 item responses (0-100 rescored) with race attached."""
    items = pd.read_excel(HSI_ITEM_PATH)
    item_cols = [c for c in items.columns if c.startswith("PHSI")]
    for col in item_cols:
        items[col] = pd.to_numeric(items[col], errors="coerce")
    items["PARTID"] = pd.to_numeric(items["PARTID"], errors="coerce")
    items = items.dropna(subset=["PARTID"]).drop_duplicates("PARTID")

    if df is None:
        df = load_data(verbose=False)
    keys = df[["PARTID", "race_name"]]
    return keys.merge(items[["PARTID"] + item_cols], on="PARTID", how="left")


def cronbach_alpha(frame):
    """Cronbach's alpha over complete cases. Returns (alpha, n)."""
    frame = frame.dropna()
    k = frame.shape[1]
    if len(frame) < 10 or k < 2:
        return np.nan, len(frame)
    total_var = frame.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan, len(frame)
    return (k / (k - 1)) * (1 - frame.var(ddof=1).sum() / total_var), len(frame)


def feldt_test(a1, n1, a2, n2):
    """Feldt's test comparing two independent alphas. Returns two-sided p."""
    from scipy.stats import f as f_dist

    if not np.isfinite([a1, a2]).all() or a1 == 1 or a2 == 1:
        return np.nan
    w = (1 - a1) / (1 - a2)
    cdf = f_dist.cdf(w, n1 - 1, n2 - 1)
    return 2 * min(cdf, 1 - cdf)


def alpha_by_race(items=None, scales=None):
    """Alpha per HSI scale within each race, with Feldt comparison."""
    if items is None:
        items = load_items()
    if scales is None:
        scales = HSI_OUTCOMES

    rows = []
    for scale in scales:
        cols = [f"PHSI{i}" for i in RAND_ITEMS[scale]]
        result = {"hsi_scale": scale, "k_items": len(cols)}
        for group in ["White", "Black"]:
            a, n = cronbach_alpha(items.loc[items["race_name"] == group, cols])
            result[f"alpha_{group}"] = a
            result[f"n_{group}"] = n
        result["alpha_diff"] = result["alpha_White"] - result["alpha_Black"]
        result["feldt_p"] = feldt_test(
            result["alpha_White"], result["n_White"], result["alpha_Black"], result["n_Black"]
        )
        rows.append(result)
    return pd.DataFrame(rows)


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def compare_correlations(r1, n1, r2, n2):
    """Two-independent-correlations z test. Returns (z, two-sided p, Cohen's q)."""
    z1, z2 = fisher_z(r1), fisher_z(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se
    from scipy.stats import norm

    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p, abs(z1 - z2)


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR
