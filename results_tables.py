"""Publication-formatted tables for the RoBERTa results and bias analyses.

Table 6  model performance by outcome and racial test group, for the combined
         training sample (the primary analysis) alongside own-race training
Table 7  the bias analyses condensed to one row per outcome

Reads the CSVs written by rq2_results.py, training_set_comparison.py and
bias_analyses.py, so run those first.
"""

import os

import pandas as pd

from hsidata import HSI_OUTCOMES, ensure_out_dir

OUT = ensure_out_dir()


def fmt_p(v):
    if pd.isna(v):
        return ""
    return "< .001" if v < 0.001 else f"{v:.3f}".lstrip("0")


def need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}\nRun the analysis scripts first.")
    return pd.read_csv(path)


def order(frame, col="HSI outcome"):
    rank = {o: i for i, o in enumerate(HSI_OUTCOMES)}
    return frame.sort_values(col, key=lambda s: s.map(rank))


def table6():
    """Model performance: combined training, then own-race training."""
    perf = need("rq2_model_performance_by_race.csv")
    own = need("training_set_comparison.csv")

    rows = []
    for _, r in order(perf).iterrows():
        outcome = r["HSI outcome"]
        block = own[own["HSI outcome"] == outcome]

        def own_val(group, field):
            m = block[(block["Trained on"] == group) & (block["Tested on"] == group)]
            return round(m.iloc[0][field], 3) if len(m) else ""

        rows.append({
            "HSI outcome": outcome,
            "N_White": int(r["n_White"]), "R2_White": r["R2_White"], "r_White": r["r_White"],
            "N_Black": int(r["n_Black"]), "R2_Black": r["R2_Black"], "r_Black": r["r_Black"],
            "z": r["z_accuracy"], "p": fmt_p(r["p_accuracy"]),
            "R2_White_own": own_val("White", "R2"), "r_White_own": own_val("White", "r"),
            "R2_Black_own": own_val("Black", "R2"), "r_Black_own": own_val("Black", "r"),
        })
    return pd.DataFrame(rows)


def table7():
    """Bias analyses, one row per outcome."""
    bias = need("bias_mean_bias.csv")
    perf = need("bias_performance.csv")
    direc = need("bias_directional.csv")
    calib = need("bias_calibration.csv")

    merged = (bias[["HSI outcome", "bias_White", "bias_Black", "p_diff"]]
              .merge(perf[["HSI outcome", "delta_MAE", "p_permutation", "cohens_d"]], on="HSI outcome")
              .merge(direc[["HSI outcome", "b1_race", "p", "partial_R2"]], on="HSI outcome")
              .merge(calib[["HSI outcome", "slope_White", "slope_Black"]], on="HSI outcome"))
    merged = order(merged).rename(columns={
        "p_diff": "p_bias", "p": "p_b1", "b1_race": "b1", "partial_R2": "pR2",
    })
    for c in ["p_bias", "p_permutation", "p_b1"]:
        merged[c] = merged[c].map(fmt_p)
    return merged[["HSI outcome", "bias_White", "bias_Black", "p_bias",
                   "delta_MAE", "p_permutation", "cohens_d",
                   "b1", "p_b1", "pR2", "slope_White", "slope_Black"]]


def to_markdown(table, title, subtitle, note):
    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join([f"**{title}**", f"*{subtitle}*", "", header, sep, *body, "", note])


T6_NOTE = (
    "Note. Models were trained on the combined sample (N = 900) unless the column is marked "
    "'own', which indicates training within the participant's own racial sample (n = 450 each). "
    "R2 and r are computed on pooled held-out predictions from five folds. z tests the "
    "difference between the White and Black correlations (Fisher r-to-z); no difference "
    "reached p < .05."
)

T7_NOTE = (
    "Note. Error is defined as true minus predicted, so a positive mean bias indicates "
    "UNDERprediction. p_bias tests whether mean bias differs between racial groups. "
    "delta_MAE is mean absolute error for Black minus White participants, tested by "
    "permutation (10,000 resamples). b1 is the race coefficient from Error = b0 + b1(race) + "
    "b2(true score), where positive values indicate underprediction for Black participants at "
    "equivalent true scores; pR2 is its partial R2. Calibration slopes regress true on "
    "predicted scores within each group; 1.0 indicates perfect calibration."
)


def main():
    t6, t7 = table6(), table7()
    pd.set_option("display.width", 250, "display.max_columns", 30)

    print("=" * 118)
    print("TABLE 6. RoBERTa Model Performance by HSI Outcome and Racial Test Group")
    print("=" * 118)
    print(t6.to_string(index=False))
    print("\n" + T6_NOTE)

    print("\n" + "=" * 118)
    print("TABLE 7. Bias Analyses by HSI Outcome")
    print("=" * 118)
    print(t7.to_string(index=False))
    print("\n" + T7_NOTE)

    for name, tbl, title, sub, note in [
        ("table6_model_performance", t6,
         "Table 6", "RoBERTa Model Performance by HSI Outcome and Racial Test Group", T6_NOTE),
        ("table7_bias_analyses", t7,
         "Table 7", "Bias Analyses by HSI Outcome", T7_NOTE),
    ]:
        tbl.to_csv(f"{OUT}/{name}.csv", index=False)
        with open(f"{OUT}/{name}.md", "w") as fh:
            fh.write(to_markdown(tbl, title, sub, note))
    print(f"\nSaved Tables 6 and 7 to {OUT}/")


if __name__ == "__main__":
    main()
