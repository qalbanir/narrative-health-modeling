"""Supplemental Table 1 in the layout Tu used for the FFM paper.

For each HSI scale, lists the LIWC categories that correlate significantly
(p < .01) with it, split by race and by direction of the correlation. Mirrors
the structure of the personality-trait supplemental table: White block on the
left, Black block on the right, positive and negative correlations in separate
columns, ordered by descending |r|.
"""

import pandas as pd
from scipy.stats import pearsonr

from hsidata import HSI_OUTCOMES, LIWC_VARS, ensure_out_dir, load_data

ALPHA = 0.05
GROUPS = ["White", "Black"]

COLUMNS = [
    "HSI scale",
    "White Positive", "r_WP",
    "White Negative", "r_WN",
    "Black Positive", "r_BP",
    "Black Negative", "r_BN",
]

NOTE = (
    f"Note. All correlations significant at p < {f'{ALPHA:.2f}'.lstrip('0')}. "
    "N = 900 (450 Black, 450 White). LIWC categories are listed under the race and "
    "direction in which they reached significance, ordered by descending |r|. Higher HSI "
    "scores indicate better health, so a negative correlation means more of that language "
    "accompanies worse health. emo_neg = Negative emotion, emo_anx = Anxiety, "
    "emo_sad = Sadness, emo_pos = Positive emotion, tone_pos = Positive tone."
)


def significant_by_direction(df):
    """{scale: {group: {'pos': [(var, r)], 'neg': [(var, r)]}}} at p < ALPHA."""
    samples = {g: df[df["race_name"] == g] for g in GROUPS}
    out = {}
    for scale in HSI_OUTCOMES:
        out[scale] = {g: {"pos": [], "neg": []} for g in GROUPS}
        for group in GROUPS:
            for var in LIWC_VARS:
                clean = samples[group][[var, scale]].dropna()
                if len(clean) < 3:
                    continue
                r, p = pearsonr(clean[var], clean[scale])
                if p < ALPHA:
                    out[scale][group]["pos" if r > 0 else "neg"].append((var, r))
        for group in GROUPS:
            for sign in ["pos", "neg"]:
                out[scale][group][sign].sort(key=lambda pair: -abs(pair[1]))
    return out


def build(significant):
    rows = []
    for scale in HSI_OUTCOMES:
        blocks = [
            significant[scale]["White"]["pos"],
            significant[scale]["White"]["neg"],
            significant[scale]["Black"]["pos"],
            significant[scale]["Black"]["neg"],
        ]
        height = max(len(b) for b in blocks)
        if height == 0:
            rows.append(dict(zip(COLUMNS, [scale] + [""] * 8)))
            continue
        for i in range(height):
            cells = []
            for block in blocks:
                if i < len(block):
                    var, r = block[i]
                    cells += [var, f"{r:.2f}"]
                else:
                    cells += ["", ""]
            rows.append(dict(zip(COLUMNS, [scale if i == 0 else ""] + cells)))
    return pd.DataFrame(rows, columns=COLUMNS)


def to_markdown(table):
    header = "| " + " | ".join(
        ["HSI scale", "Positive", "r", "Negative", "r", "Positive", "r", "Negative", "r"]
    ) + " |"
    sub = "| | *White* | | | | *Black* | | | |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join(
        ["**Supplemental Table 1**",
         "*Correlations Between LIWC Scores and HSI Scores Across Races*",
         "", header, sub, sep, *body, "", NOTE]
    )


def main():
    out_dir = ensure_out_dir()
    df = load_data(verbose=False)
    significant = significant_by_direction(df)
    table = build(significant)

    pd.set_option("display.width", 220, "display.max_rows", 200)
    print("\nSupplemental Table 1")
    print("Correlations Between LIWC Scores and HSI Scores Across Races")
    print("=" * 118)
    print(table.to_string(index=False))
    print("\n" + NOTE)

    counts = {
        g: sum(len(significant[s][g][d]) for s in HSI_OUTCOMES for d in ["pos", "neg"])
        for g in GROUPS
    }
    print(f"\nSignificant at p < {ALPHA}:  White {counts['White']}  |  Black {counts['Black']}")

    table.to_csv(f"{out_dir}/supplemental_table1.csv", index=False)
    with open(f"{out_dir}/supplemental_table1.md", "w") as fh:
        fh.write(to_markdown(table))
    print(f"\nSaved:\n  {out_dir}/supplemental_table1.csv\n  {out_dir}/supplemental_table1.md")


if __name__ == "__main__":
    main()
