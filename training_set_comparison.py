"""Compare models trained on the combined, White, and Black samples.

Mirrors the structure Tu reports for the FFM paper: for each HSI outcome, how
well does each training set predict each racial test group?

What the runs give us
---------------------
Each script does 5-fold CV inside its own training sample, so the held-out
predictions only cover the people in that sample:

    combined  -> test predictions for all 900, splittable by race
    White     -> test predictions for the 450 White participants
    Black     -> test predictions for the 450 Black participants

That yields four of the six cells in Tu's grid. The two cross-race transfer
cells -- White-trained predicting Black participants, and the reverse -- need a
saved fold model applied to the other sample, which is a separate step and not
available from these runs. Cells that do not exist are printed as "--" rather
than left ambiguous.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from hsidata import LIWC_PATH, RACE_MAP, compare_correlations, ensure_out_dir
from rq2_results import OUTCOMES as COMBINED_STEMS

RUNS_DIR = "/path/to/project/roberta_runs"
GROUPS = ["White", "Black"]
N_FOLDS = 5

TRAINING_SETS = {"c": "Combined", "w": "White", "b": "Black"}

# outcome -> short code, inverted from rq2_results.OUTCOMES ("phcpsmc" -> "phc")
CODES = {outcome: stem[:-4] for stem, outcome in COMBINED_STEMS.items()}


def load_race():
    race = pd.read_csv(LIWC_PATH, low_memory=False, usecols=["PARTID", "PRACE"])
    race["race_name"] = race["PRACE"].map(RACE_MAP)
    return race[race["race_name"].isin(GROUPS)]


def pooled(stem):
    files = sorted(glob.glob(os.path.join(RUNS_DIR, stem, "Model*", "ffn_test_predictions_fold*.csv")))
    if len(files) < N_FOLDS:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def cell(df, group):
    """Metrics for one training set evaluated on one racial group."""
    s = df[df["race_name"] == group]
    if len(s) < 10:
        return None
    r, _ = pearsonr(s["true_label"], s["predicted_label"])
    return {
        "n": len(s),
        "r": round(r, 3),
        "R2": round(r2(s["true_label"], s["predicted_label"]), 3),
        "MAE": round((s["predicted_label"] - s["true_label"]).abs().mean(), 3),
        "_r_raw": r,
    }


def main():
    out_dir = ensure_out_dir()
    if not os.path.isdir(RUNS_DIR):
        raise SystemExit(f"Cannot reach {RUNS_DIR}\nIs the ACCOUNT share mounted?")

    race = load_race()
    rows, pending = [], []

    for outcome, code in CODES.items():
        for suffix, train_label in TRAINING_SETS.items():
            stem = f"{code}psm{suffix}"
            preds = pooled(stem)
            if preds is None:
                pending.append(stem)
                continue
            df = preds.merge(race, on="PARTID", how="inner")
            for group in GROUPS:
                metrics = cell(df, group)
                if metrics is None:
                    continue
                rows.append({
                    "HSI outcome": outcome,
                    "Trained on": train_label,
                    "Tested on": group,
                    **{k: v for k, v in metrics.items() if not k.startswith("_")},
                })

    if not rows:
        print("No completed runs found yet.")
        if pending:
            print(f"Waiting on {len(pending)} run(s).")
        return

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", 300)

    print("=" * 92)
    print("Model performance by training sample and test group")
    print("=" * 92)
    for outcome in CODES:
        block = results[results["HSI outcome"] == outcome]
        if block.empty:
            continue
        print(f"\n{outcome}")
        pivot = block.pivot(index="Trained on", columns="Tested on", values=["n", "r", "R2", "MAE"])
        pivot = pivot.reindex(index=[t for t in TRAINING_SETS.values() if t in pivot.index])
        print(pivot.to_string())

    # Does training on your own race help? Compare within-race CV to the
    # combined model evaluated on that same race.
    print("\n" + "=" * 92)
    print("Own-race training vs combined training, same test group")
    print("=" * 92)
    comp = []
    for outcome in CODES:
        for group in GROUPS:
            own = results[(results["HSI outcome"] == outcome) & (results["Trained on"] == group)
                          & (results["Tested on"] == group)]
            comb = results[(results["HSI outcome"] == outcome) & (results["Trained on"] == "Combined")
                           & (results["Tested on"] == group)]
            if own.empty or comb.empty:
                continue
            r_own, n_own = own.iloc[0]["r"], own.iloc[0]["n"]
            r_comb, n_comb = comb.iloc[0]["r"], comb.iloc[0]["n"]
            z, p, _ = compare_correlations(r_own, n_own, r_comb, n_comb)
            comp.append({
                "HSI outcome": outcome, "Test group": group,
                f"r_own": r_own, "r_combined": r_comb,
                "diff": round(r_own - r_comb, 3), "z": round(z, 2), "p": round(p, 3),
            })
    if comp:
        print(pd.DataFrame(comp).to_string(index=False))
        print("\nNote. Positive diff means the own-race model beat the combined model.")
        print("These share test cases, so the z test is approximate -- read it as descriptive.")

    path = f"{out_dir}/training_set_comparison.csv"
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    if pending:
        print(f"\nStill incomplete ({len(pending)}): {', '.join(sorted(pending))}")
    print("\nCross-race transfer (White-trained -> Black, and reverse) is not in these")
    print("runs; it needs saved fold models applied to the other sample.")


if __name__ == "__main__":
    main()
