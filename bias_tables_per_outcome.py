"""One bias table per HSI outcome, in the four-block layout used for the
personality models: accuracy, performance bias, directional bias, calibration.

Reads the CSVs written by bias_analyses.py and reshapes them; no statistics are
recomputed here except Levene's degrees of freedom, which the summary tables do
not carry.
"""

import os

import pandas as pd

from hsidata import HSI_OUTCOMES, ensure_out_dir

OUT = ensure_out_dir()


def fmt_p(v):
    if pd.isna(v):
        return ""
    return "< .001" if v < 0.001 else f"{v:.3f}".lstrip("0")


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run bias_analyses.py first.")
    return pd.read_csv(path).set_index("HSI outcome")


def build(outcome, acc, bias, dist, perf, direc, calib):
    a, b, d = acc.loc[outcome], bias.loc[outcome], dist.loc[outcome]
    p, dr, c = perf.loc[outcome], direc.loc[outcome], calib.loc[outcome]
    n_total = int(a["n_White"] + a["n_Black"])

    lines = []
    lines.append(["Accuracy", "RMSE", "R2", "r", "Mean Bias", "Mean_true", "Mean_pred",
                  "Var_true", "Var_pred", "F", "df1", "df2", "p"])
    for g in ["White", "Black"]:
        row = [g, a[f"RMSE_{g}"], a[f"R2_{g}"], a[f"r_{g}"], b[f"bias_{g}"],
               d[f"true_M_{g}"], d[f"pred_M_{g}"], d[f"true_Var_{g}"], d[f"pred_Var_{g}"]]
        # Levene is one test for the pair, reported on the first row
        row += ([round(d["levene_W"], 2), 1, n_total - 2, fmt_p(d["levene_p"])]
                if g == "White" else ["", "", "", ""])
        lines.append(row)

    lines.append([])
    lines.append(["Performance Bias", "MAE", "delta_MAE", "p", "Cohen's d"])
    lines.append(["White", p["MAE_White"], p["delta_MAE"], fmt_p(p["p_permutation"]), p["cohens_d"]])
    lines.append(["Black", p["MAE_Black"], "", "", ""])

    lines.append([])
    lines.append(["Directional Bias", "b1", "p", "partial R2"])
    lines.append(["", dr["b1_race"], fmt_p(dr["p"]), round(dr["partial_R2"], 3)])

    lines.append([])
    lines.append(["Calibration", "Slope", "p"])
    for g in ["White", "Black"]:
        lines.append([g, c[f"slope_{g}"], fmt_p(c[f"p_vs1_{g}"])])
    return lines


def render(lines):
    width = max(len(r) for r in lines if r)
    padded = [r + [""] * (width - len(r)) if r else [""] * width for r in lines]
    return pd.DataFrame(padded)


def main():
    acc, bias = load("bias_accuracy.csv"), load("bias_mean_bias.csv")
    dist, perf = load("bias_distribution.csv"), load("bias_performance.csv")
    direc, calib = load("bias_directional.csv"), load("bias_calibration.csv")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    all_frames = []

    for i, outcome in enumerate(HSI_OUTCOMES, start=6):
        lines = build(outcome, acc, bias, dist, perf, direc, calib)
        frame = render(lines)
        all_frames.append((outcome, i, frame))

        print("\n" + "=" * 96)
        print(f"Table {i}. Bias Analyses of the {outcome} Model")
        print("=" * 96)
        print(frame.to_string(index=False, header=False))

        slug = outcome.lower().replace(" ", "_").replace("/", "_")
        frame.to_csv(f"{OUT}/biastable_{slug}.csv", index=False, header=False)

    print(f"\nWrote {len(all_frames)} per-outcome tables to {OUT}/biastable_*.csv")
    print("Table numbers above assume the bias tables begin at Table 6; adjust to your ordering.")


if __name__ == "__main__":
    main()
