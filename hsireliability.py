"""Internal consistency of the Health Status Inventory by race.

Unequal reliability across groups attenuates correlations unequally, which can
imitate a genuine racial difference in how language relates to health. This
checks that assumption before the correlation results are interpreted, and
confirms the scored subscale file reproduces from the item-level responses.
"""

import numpy as np
import pandas as pd

from hsidata import (
    HSI_OUTCOMES,
    HSI_SUBSCALES,
    RAND_ITEMS,
    alpha_by_race,
    ensure_out_dir,
    load_data,
    load_items,
)

ALPHA = 0.01


def verify_scoring(df, items):
    """Confirm RAND item groupings reproduce the scored subscale file."""
    merged = items.merge(df[["PARTID"] + HSI_SUBSCALES], on="PARTID", how="inner")
    rows = []
    for scale in HSI_SUBSCALES:
        cols = [f"PHSI{i}" for i in RAND_ITEMS[scale]]
        calc = merged[cols].mean(axis=1)
        filed = merged[scale]
        ok = calc.notna() & filed.notna()
        rows.append(
            {
                "hsi_scale": scale,
                "n_compared": int(ok.sum()),
                "max_abs_diff": round(float((calc[ok] - filed[ok]).abs().max()), 6),
                "r": round(float(calc[ok].corr(filed[ok])), 6),
            }
        )
    return pd.DataFrame(rows)


def main():
    out_dir = ensure_out_dir()
    df = load_data()
    items = load_items(df)

    check = verify_scoring(df, items)
    reliability = alpha_by_race(items, HSI_OUTCOMES)

    pd.set_option("display.width", 200, "display.max_columns", 30)

    print("\n" + "=" * 92)
    print("SCORING CHECK: item-derived subscales vs. the scored subscale file")
    print("=" * 92)
    print(check.to_string(index=False))
    if check["max_abs_diff"].max() < 1e-6:
        print("\nAll eight subscales reproduce exactly. The scored file is correct.")

    print("\n" + "=" * 92)
    print("CRONBACH'S ALPHA BY RACE")
    print(f"feldt_p < {ALPHA} flags unequal reliability across groups.")
    print("=" * 92)
    printable = reliability.copy()
    for col in ["alpha_White", "alpha_Black", "alpha_diff"]:
        printable[col] = printable[col].round(3)
    printable["feldt_p"] = printable["feldt_p"].map(lambda v: f"{v:.4f}")
    printable["flag"] = np.where(reliability["feldt_p"] < ALPHA, "UNEQUAL", "")
    print(printable.to_string(index=False))

    unequal = reliability[reliability["feldt_p"] < ALPHA]
    print("\n" + "=" * 92)
    print("INTERPRETATION")
    print("=" * 92)
    if unequal.empty:
        print(f"Reliability is statistically equivalent across race on all {len(reliability)} scales.")
        print("Correlation comparisons are not confounded by differential attenuation.")
    else:
        print(f"{len(unequal)} of {len(reliability)} scales show unequal reliability:")
        for _, row in unequal.iterrows():
            lower = "Black" if row["alpha_diff"] > 0 else "White"
            print(f"  {row['hsi_scale']}: alpha_White={row['alpha_White']:.3f}, "
                  f"alpha_Black={row['alpha_Black']:.3f} -- lower in the {lower} sample")
        print("\nCorrelations with these scales are attenuated more in the lower-reliability")
        print("group. Race differences involving them need the disattenuated sensitivity")
        print("check in hsiliwccorrelation.py before being read as substantive.")

    path = f"{out_dir}/hsi_reliability_by_race.csv"
    check_path = f"{out_dir}/hsi_scoring_verification.csv"
    reliability.to_csv(path, index=False)
    check.to_csv(check_path, index=False)
    print(f"\nSaved:\n  {path}\n  {check_path}")


if __name__ == "__main__":
    main()
