"""Test whether the calibration slope itself differs by race.

bias_analyses.py tests each group's calibration slope against 1, which answers
"is this group's prediction on the right scale?" but not "is the scale the same
for both groups?" -- the question every other bias component asks. That comparison
is the race x prediction interaction:

    true = b0 + b1*predicted + b2*black + b3*(predicted x black)

b3 is the difference in calibration slope, Black minus White. A negative b3 means
predictions are more compressed for Black participants: the same one-unit change
in the model's output corresponds to a smaller change in observed health.

Reads the same pooled fold predictions bias_analyses.py uses, so the slopes here
reproduce the ones in bias_calibration.csv.
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from bias_analyses import load_race, pooled
from hsidata import ensure_out_dir
from rq2_results import OUTCOMES

OUT = ensure_out_dir()


def main():
    race = load_race()
    rows = []
    for stem, outcome in OUTCOMES.items():
        df = pooled(stem, race)
        if df is None:
            print(f"  {outcome}: predictions incomplete, skipped")
            continue
        fit = smf.ols("true_label ~ predicted_label * black", data=df).fit()
        rows.append({
            "HSI outcome": outcome,
            "slope_White": round(fit.params["predicted_label"], 3),
            "slope_diff_B_minus_W": round(fit.params["predicted_label:black"], 3),
            "SE": round(fit.bse["predicted_label:black"], 3),
            "t": round(fit.tvalues["predicted_label:black"], 2),
            "p": fit.pvalues["predicted_label:black"],
        })

    out = pd.DataFrame(rows)
    out["q_BH"] = multipletests(out["p"], method="fdr_bh")[1]

    pd.set_option("display.width", 180)
    print("=" * 92)
    print("CALIBRATION SLOPE DIFFERENCE (Black minus White)")
    print("=" * 92)
    print(out.assign(p=out["p"].map(lambda v: f"{v:.4f}"),
                     q_BH=out["q_BH"].map(lambda v: f"{v:.4f}")).to_string(index=False))
    print(f"\n{(out['q_BH'] < .05).sum()} of {len(out)} differ after BH correction")
    print(f"slope_diff negative in {(out['slope_diff_B_minus_W'] < 0).sum()} of {len(out)}")

    out.to_csv(f"{OUT}/bias_calibration_difference.csv", index=False)
    print(f"\nSaved: {OUT}/bias_calibration_difference.csv")


if __name__ == "__main__":
    main()
