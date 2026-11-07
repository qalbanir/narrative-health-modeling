"""Build the RoBERTa fine-tuning input files for the HSI regression task.

finetune.py expects a JSON-lines file with PARTID, text, PRACE, and the label
column. This writes that file for the Health Status Inventory outcomes, in the
three training samples the methods call for: combined, Black only, White only.

Run locally, then copy the resulting files to the GPU cluster.
"""

import os

import pandas as pd

from hsidata import (
    HSI_COMPOSITES,
    HSI_OUTCOMES,
    HSI_SUBSCALES,
    LIWC_PATH,
    ensure_out_dir,
    load_data,
)

MODEL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data")

SAMPLES = {
    "combined": None,
    "black": "Black",
    "white": "White",
}


def build_frame():
    """Merge narrative text onto the HSI + race frame."""
    hsi = load_data(verbose=False)
    text = pd.read_csv(LIWC_PATH, low_memory=False, usecols=["PARTID", "text"])

    df = hsi[["PARTID", "PRACE", "race_name"] + HSI_OUTCOMES].merge(text, on="PARTID", how="inner")
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["n_words"] = df["text"].str.split().str.len()
    return df


def main():
    os.makedirs(MODEL_DATA_DIR, exist_ok=True)
    ensure_out_dir()
    df = build_frame()

    columns = ["PARTID", "PRACE", "race_name", "text"] + HSI_OUTCOMES

    print(f"Narratives with text: {len(df)}")
    print(f"Words per narrative: median {df['n_words'].median():.0f}, "
          f"max {df['n_words'].max():.0f}")
    print()

    written = []
    for name, race in SAMPLES.items():
        subset = df if race is None else df[df["race_name"] == race]
        path = os.path.join(MODEL_DATA_DIR, f"hsi_{name}.json")
        subset[columns].to_json(path, orient="records", lines=True)
        written.append((name, len(subset), path))
        print(f"{name:9s} n={len(subset):4d} -> {path}")

    print("\nLabel availability per outcome (combined sample):")
    counts = pd.DataFrame(
        {
            "outcome": HSI_OUTCOMES,
            "type": ["Composite" if o in HSI_COMPOSITES else "Subscale" for o in HSI_OUTCOMES],
            "n_labeled": [int(df[o].notna().sum()) for o in HSI_OUTCOMES],
            "n_missing": [int(df[o].isna().sum()) for o in HSI_OUTCOMES],
        }
    )
    print(counts.to_string(index=False))

    print(f"\nSubscales: {len(HSI_SUBSCALES)} | Composites: {len(HSI_COMPOSITES)}")
    print(f"Files written to {MODEL_DATA_DIR}/")
    print("\nCopy these to the cluster, then point finetune.py's read_json() at one of them.")


if __name__ == "__main__":
    main()
