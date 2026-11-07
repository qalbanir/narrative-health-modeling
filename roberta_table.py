"""Table 5 equivalent: RoBERTa language modeling of health status.

Reproduces the layout used for the personality paper -- for each HSI outcome, a
row per training sample (White, Black, combined) and a column block per test
group (White, Black, combined), each reporting MSE, r and R2.

Cells come from three places:

    combined training      the held-out folds of <code>psmc, split by race
    own-race training      the held-out folds of <code>psmw / <code>psmb
    cross-race training    crossrace_* predictions written by cross_race_predict.py

Cells whose source has not finished are left blank, so the table can be built at
any point and will fill in as runs complete.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from hsidata import HSI_OUTCOMES, LIWC_PATH, RACE_MAP, ensure_out_dir
from rq2_results import OUTCOMES, RUNS_DIR

N_FOLDS = 5
TRAIN_ROWS = [("w", "White training set"), ("b", "Black training set"), ("c", "Combined training set")]
TEST_COLS = ["White", "Black", "Combined"]

CODES = {outcome: stem[:-4] for stem, outcome in OUTCOMES.items()}


def load_race():
    race = pd.read_csv(LIWC_PATH, low_memory=False, usecols=["PARTID", "PRACE"])
    race["race_name"] = race["PRACE"].map(RACE_MAP)
    return race[race["race_name"].isin(["White", "Black"])]


def read_folds(pattern):
    files = sorted(glob.glob(pattern))
    if len(files) < N_FOLDS:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def predictions(code, train_suffix, race):
    """All available held-out predictions for one training sample, race-labelled."""
    stem = f"{code}psm{train_suffix}"
    model_dir = os.path.join(RUNS_DIR, stem, f"Model{stem}")

    own = read_folds(os.path.join(model_dir, "ffn_test_predictions_fold*.csv"))
    frames = []
    if own is not None:
        frames.append(own.merge(race, on="PARTID", how="inner"))

    # cross-race predictions, if that transfer has been run
    for other in ["white", "black"]:
        cross = read_folds(os.path.join(model_dir, f"crossrace_{other}",
                                        "crossrace_test_predictions_fold*.csv"))
        if cross is not None:
            frames.append(cross.merge(race, on="PARTID", how="inner"))

    return pd.concat(frames, ignore_index=True) if frames else None


def metrics(df):
    if df is None or len(df) < 10:
        return {"MSE": "", "r": "", "R2": ""}
    y, yhat = df["true_label"], df["predicted_label"]
    r, _ = pearsonr(y, yhat)
    return {
        "MSE": f"{((y - yhat) ** 2).mean():.2f}",
        "r": f"{r:.2f}",
        "R2": f"{1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum():.2f}",
    }


def build():
    race = load_race()
    rows = []
    for outcome in HSI_OUTCOMES:
        code = CODES[outcome]
        rows.append({"Outcome / training set": outcome,
                     **{f"{t}_{m}": m for t in TEST_COLS for m in ["MSE", "r", "R2"]}})
        for suffix, label in TRAIN_ROWS:
            preds = predictions(code, suffix, race)
            row = {"Outcome / training set": label}
            for test in TEST_COLS:
                if preds is None:
                    cell = metrics(None)
                elif test == "Combined":
                    # only meaningful when the model has seen, and been tested on, both
                    both = preds if preds["race_name"].nunique() == 2 else None
                    cell = metrics(both)
                else:
                    cell = metrics(preds[preds["race_name"] == test])
                for m, v in cell.items():
                    row[f"{test}_{m}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def sample_note(race):
    parts = []
    for outcome in HSI_OUTCOMES:
        stem = f"{CODES[outcome]}psmc"
        preds = read_folds(os.path.join(RUNS_DIR, stem, f"Model{stem}",
                                        "ffn_test_predictions_fold*.csv"))
        if preds is None:
            continue
        merged = preds.merge(race, on="PARTID", how="inner")
        n_w = (merged["race_name"] == "White").sum()
        n_b = (merged["race_name"] == "Black").sum()
        parts.append(f"{outcome} N_White = {n_w}, N_Black = {n_b}")
    return "Note. " + "; ".join(parts) + "."


def main():
    out_dir = ensure_out_dir()
    if not os.path.isdir(RUNS_DIR):
        raise SystemExit(f"Cannot reach {RUNS_DIR}\nIs the ACCOUNT share mounted?")

    race = load_race()
    table = build()
    note = sample_note(race)

    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_rows", 60)
    print("=" * 110)
    print("TABLE 5. RoBERTa Language Modeling of Health Status")
    print("=" * 110)
    print(table.to_string(index=False))

    blanks = (table.drop(columns=["Outcome / training set"]) == "").sum().sum()
    if blanks:
        print(f"\n{blanks} empty cells -- the corresponding cross-race transfers have not finished.")

    path = f"{out_dir}/table5_roberta_health_modeling.csv"
    table.to_csv(path, index=False)
    with open(f"{out_dir}/table5_roberta_health_modeling_note.txt", "w") as fh:
        fh.write(note + "\n")

    md_path = f"{out_dir}/table5_roberta_health_modeling.md"
    with open(md_path, "w") as fh:
        fh.write(to_markdown(table, note))
    print(f"\nSaved:\n  {path}\n  {md_path}\n  {out_dir}/table5_roberta_health_modeling_note.txt")


def to_markdown(table, note):
    """Paste-ready table: outcome names become section headers, as in the source layout."""
    header = ("| Training set | MSE | r | R2 | MSE | r | R2 | MSE | r | R2 |\n"
              "| | *White test* | | | *Black test* | | | *Combined test* | | |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    lines = ["**Table 5**", "*RoBERTa Language Modeling of Health Status*", ""]
    for _, row in table.iterrows():
        label = row["Outcome / training set"]
        values = [row[f"{t}_{m}"] for t in TEST_COLS for m in ["MSE", "r", "R2"]]
        if values[0] == "MSE":          # outcome header row
            lines += ["", f"**{label}**", "", header]
        else:
            lines.append("| " + label + " | " + " | ".join(str(v) for v in values) + " |")
    lines += ["", note]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
