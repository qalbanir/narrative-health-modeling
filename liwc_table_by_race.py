"""Pooled and by-race LIWC-HSI correlations for the twelve selected categories,
with significance markers, in a layout that spells the HSI subscales out in full.

Reads the correlations already computed by hsiliwccorrelation.py; nothing is
recomputed. Writes a CSV and a paste-ready markdown version.
"""

import pandas as pd

from hsidata import HSI_OUTCOMES, LIWC_VARS, ensure_out_dir

OUT = ensure_out_dir()


def star(r, p):
    if pd.isna(r):
        return ""
    mark = "**" if p < .01 else "*" if p < .05 else ""
    return f"{r:+.2f}".replace("0.", ".") + mark


def build():
    long = pd.read_csv(f"{OUT}/liwc_hsi_correlations_by_race.csv")
    pooled_r = pd.read_csv(f"{OUT}/liwc_hsi_correlation_matrix_pooled_r.csv", index_col=0)
    pooled_p = pd.read_csv(f"{OUT}/liwc_hsi_correlation_matrix_pooled_p.csv", index_col=0)
    idx = long.set_index(["hsi_scale", "liwc_variable"])

    rows = []
    for outcome in HSI_OUTCOMES:
        for group in ["Combined", "White", "Black"]:
            row = {"HSI subscale": outcome if group == "Combined" else "", "Sample": group}
            for var in LIWC_VARS:
                if group == "Combined":
                    row[var] = star(pooled_r.loc[var, outcome], pooled_p.loc[var, outcome])
                else:
                    rec = idx.loc[(outcome, var)]
                    key = group.lower()
                    row[var] = star(rec[f"{key}_r"], rec[f"{key}_p"])
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table):
    cols = list(table.columns)
    lines = ["**Supplemental Table**",
             "*Correlations Between Selected LIWC Categories and HSI Scores, "
             "Overall and by Racial Group*", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in table.iterrows():
        label = r["HSI subscale"]
        cells = [f"**{label}**" if label else ""] + [str(r[c]) for c in cols[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "*Note.* N = 893 to 894 overall (447 to 448 White, 444 to 446 Black). "
              "Correlations are Pearson r between the LIWC category and the HSI score. "
              "\\* p < .05, \\*\\* p < .01, uncorrected. Racial differences in these "
              "correlations were tested with Fisher's r-to-z; none survived "
              "Benjamini-Hochberg correction across the 120 comparisons."]
    return "\n".join(lines)


def main():
    table = build()
    pd.set_option("display.width", 260, "display.max_columns", 20, "display.max_rows", 40)
    print(table.to_string(index=False))
    table.to_csv(f"{OUT}/liwc_hsi_by_race_table.csv", index=False)
    with open(f"{OUT}/liwc_hsi_by_race_table.md", "w") as fh:
        fh.write(to_markdown(table))
    print(f"\nSaved:\n  {OUT}/liwc_hsi_by_race_table.csv\n  {OUT}/liwc_hsi_by_race_table.md")


if __name__ == "__main__":
    main()
