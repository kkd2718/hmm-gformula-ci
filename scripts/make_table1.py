"""Table 1: baseline characteristics, by Berlin severity strata.

Outputs publication-ready markdown table with median (IQR) for continuous
and N (%) for categorical.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def med_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "—"
    m = np.median(x); q1, q3 = np.percentile(x, [25, 75])
    return f"{m:.1f} ({q1:.1f}-{q3:.1f})"


def n_pct(mask):
    n = int(mask.sum()); tot = len(mask)
    return f"{n} ({100*n/tot:.1f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    args = p.parse_args()
    df = pd.read_csv(args.csv)

    # First-day per stay
    by_stay = df.sort_values(["stay_id", "day_num"]).groupby("stay_id").first().reset_index()
    sev = by_stay.get("severity", pd.Series(["unknown"] * len(by_stay)))
    print(f"Unique stays = {len(by_stay)}")
    print("Severity distribution:")
    print(sev.value_counts())

    md = ["# Table 1. Baseline characteristics", "",
          f"_N = {len(by_stay)} ICU stays. Continuous: median (IQR). "
          "Categorical: N (%). Stratified by Berlin ARDS severity._", ""]
    md.append("| Characteristic | Overall | Mild | Moderate | Severe |")
    md.append("|---|---|---|---|---|")

    strata = {"Overall": pd.Series(True, index=by_stay.index)}
    for s in ["mild", "moderate", "severe"]:
        strata[s.capitalize()] = sev == s

    rows = [
        ("N", lambda m: f"{int(m.sum())}"),
        ("Age (yr)", lambda m: med_iqr(by_stay.loc[m, "anchor_age"])),
        ("Male, N (%)", lambda m: n_pct((by_stay.loc[m, "gender"] == "M") if "gender" in by_stay.columns
                                        else by_stay.loc[m, "gender_M"] == 1 if "gender_M" in by_stay.columns
                                        else pd.Series(False, index=by_stay.loc[m].index))),
        ("BMI", lambda m: med_iqr(by_stay.loc[m, "bmi_imputed"]) if "bmi_imputed" in by_stay.columns else "—"),
        ("Charlson index", lambda m: med_iqr(by_stay.loc[m, "charlson_index"])
                                       if "charlson_index" in by_stay.columns else "—"),
        ("PaO₂/FiO₂ (day 1)", lambda m: med_iqr(by_stay.loc[m, "pf_ratio"])
                                          if "pf_ratio" in by_stay.columns else "—"),
        ("PaCO₂ (day 1)", lambda m: med_iqr(by_stay.loc[m, "paco2"])
                                      if "paco2" in by_stay.columns else "—"),
        ("Lactate (day 1)", lambda m: med_iqr(by_stay.loc[m, "lactate"])
                                        if "lactate" in by_stay.columns else "—"),
        ("Heart rate (day 1)", lambda m: med_iqr(by_stay.loc[m, "heart_rate"])
                                            if "heart_rate" in by_stay.columns else "—"),
        ("MAP (day 1)", lambda m: med_iqr(by_stay.loc[m, "map_mmhg"])
                                    if "map_mmhg" in by_stay.columns else "—"),
        ("GCS total (day 1)", lambda m: med_iqr(by_stay.loc[m, "gcs_total"])
                                          if "gcs_total" in by_stay.columns else "—"),
        ("Creatinine (day 1)", lambda m: med_iqr(by_stay.loc[m, "creatinine"])
                                            if "creatinine" in by_stay.columns else "—"),
        ("Mechanical power (day 1, J/min)", lambda m: med_iqr(by_stay.loc[m, "mp_j_min"])
                                                  if "mp_j_min" in by_stay.columns else "—"),
        ("30-day mortality, N (%)",
         lambda m: n_pct((by_stay.loc[m, "mortality_30d"] == 1).fillna(False))
                   if "mortality_30d" in by_stay.columns else "—"),
    ]
    for label, fn in rows:
        cells = [fn(strata[s]) for s in ["Overall", "Mild", "Moderate", "Severe"]]
        md.append(f"| {label} | " + " | ".join(cells) + " |")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
