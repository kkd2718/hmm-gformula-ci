"""Generate Table 1 (Patient characteristics stratified by ARDS severity) for the
ARDS-MP causal study, MIMIC-IV v3.1 cohort. Endpoint: 28-day cumulative mortality.

Stratifier: Berlin definition severity at day 0 (mild / moderate / severe by P/F ratio).
Tests: Kruskal-Wallis for continuous, chi-square for categorical.
Output: markdown table to Paper/학위논문/_draft/tables/table1.md
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CSV_PATH = Path(
    r"C:\Users\기덕\Desktop\Study\YMC_MPH\MR_causal\MIMIC-IV\3.1\cohort_v31\ards_v31_v4.csv"
)
OUT_PATH = Path(
    r"C:\Users\기덕\Desktop\Study\Paper\학위논문\_draft\tables\table1.md"
)
T_MAX = 28


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_n_pct(n: int, total: int) -> str:
    return f"{n:,} ({100.0 * n / max(total, 1):.1f})"


def fmt_med_iqr(values: pd.Series) -> str:
    v = values.dropna()
    if len(v) == 0:
        return "-"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def fmt_mean_sd(values: pd.Series) -> str:
    v = values.dropna()
    if len(v) == 0:
        return "-"
    return f"{v.mean():.1f} ({v.std(ddof=1):.1f})"


def kruskal_p(values: pd.Series, groups: pd.Series) -> float:
    arrs = [values[groups == g].dropna().to_numpy() for g in pd.unique(groups)]
    arrs = [a for a in arrs if len(a) > 0]
    if len(arrs) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*arrs).pvalue)
    except Exception:
        return np.nan


def chi2_p(binary: pd.Series, groups: pd.Series) -> float:
    tab = pd.crosstab(binary.fillna(0).astype(int), groups)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    try:
        return float(stats.chi2_contingency(tab.values, correction=False)[1])
    except Exception:
        return np.nan


def chi2_p_categorical(values: pd.Series, groups: pd.Series) -> float:
    tab = pd.crosstab(values, groups)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    try:
        return float(stats.chi2_contingency(tab.values, correction=False)[1])
    except Exception:
        return np.nan


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} (stay, day) rows")
    df = df[df["day_num"] < T_MAX].copy()

    first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id").first().reset_index()
    )
    death28 = df.groupby("stay_id")["death_event"].max().rename("mortality_28d")
    mp_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["mp_j_min"]
        .first().rename("mp_first_day")
    )
    mp_mean = df.groupby("stay_id")["mp_j_min"].mean().rename("mp_mean_28d")
    pf_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["pf_ratio"]
        .first().rename("pf_first_day")
    )
    peep_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["peep"]
        .first().rename("peep_first_day")
    )
    fio2_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["fio2"]
        .first().rename("fio2_first_day")
    )
    lactate_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["lactate"]
        .first().rename("lactate_first_day")
    )
    map_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["map_mmhg"]
        .first().rename("map_first_day")
    )
    vt_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["tidvol_obs"]
        .first().rename("vt_first_day")
    )
    rr_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["rr"]
        .first().rename("rr_first_day")
    )
    pplat_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["pplat"]
        .first().rename("pplat_first_day")
    )
    dp_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["driving_pressure"]
        .first().rename("dp_first_day")
    )
    sofa_first = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id")["sofa_24h"]
        .first().rename("sofa_24h")
    )

    stay = (
        first[
            [
                "stay_id", "subject_id", "anchor_age", "gender_M", "bmi_imputed",
                "charlson_index", "severity", "first_day_pf", "los",
            ]
        ]
        .merge(death28, on="stay_id")
        .merge(mp_first, on="stay_id")
        .merge(mp_mean, on="stay_id")
        .merge(pf_first, on="stay_id")
        .merge(peep_first, on="stay_id")
        .merge(fio2_first, on="stay_id")
        .merge(lactate_first, on="stay_id")
        .merge(map_first, on="stay_id")
        .merge(vt_first, on="stay_id")
        .merge(rr_first, on="stay_id")
        .merge(pplat_first, on="stay_id")
        .merge(dp_first, on="stay_id")
        .merge(sofa_first, on="stay_id")
    )
    stay = stay[stay["severity"].isin(["mild", "moderate", "severe"])].copy()
    print(
        f"Per-stay frame: {len(stay):,} stays / "
        f"{stay['subject_id'].nunique():,} unique patients"
    )

    groups_order = ["mild", "moderate", "severe"]
    sev = stay["severity"]
    overall = stay
    subsets = {g: stay[sev == g] for g in groups_order}
    counts = {g: len(s) for g, s in subsets.items()}

    rows: list[tuple[str, list[str], str]] = []

    def add_continuous(label: str, col: str, fmt=fmt_med_iqr):
        p = kruskal_p(stay[col], sev)
        rows.append((label, [fmt(overall[col])] + [fmt(subsets[g][col]) for g in groups_order], fmt_p(p)))

    def add_binary_pct(label: str, col: str):
        p = chi2_p(stay[col], sev)
        rows.append((
            label,
            [fmt_n_pct(int(overall[col].sum()), len(overall))]
            + [fmt_n_pct(int(subsets[g][col].sum()), len(subsets[g])) for g in groups_order],
            fmt_p(p),
        ))

    rows.append(("N (stays)", [f"{len(overall):,}"] + [f"{counts[g]:,}" for g in groups_order], "-"))
    rows.append((
        "N (unique patients)",
        [f"{overall['subject_id'].nunique():,}"]
        + [f"{subsets[g]['subject_id'].nunique():,}" for g in groups_order],
        "-",
    ))
    add_continuous("Age (yr), mean (SD)", "anchor_age", fmt=fmt_mean_sd)
    add_binary_pct("Male sex, n (%)", "gender_M")
    add_continuous("BMI (kg/m2), mean (SD)", "bmi_imputed", fmt=fmt_mean_sd)
    add_continuous("Charlson index, median [IQR]", "charlson_index")
    add_continuous("SOFA score (24h), median [IQR]", "sofa_24h")
    add_continuous("Day-0 P/F ratio, median [IQR]", "pf_first_day")
    add_continuous("Day-0 PEEP (cmH2O), median [IQR]", "peep_first_day")
    add_continuous("Day-0 FiO2, median [IQR]", "fio2_first_day")
    add_continuous("Day-0 tidal volume (mL), median [IQR]", "vt_first_day")
    add_continuous("Day-0 respiratory rate (bpm), median [IQR]", "rr_first_day")
    add_continuous("Day-0 plateau pressure (cmH2O), median [IQR]", "pplat_first_day")
    add_continuous("Day-0 driving pressure (cmH2O), median [IQR]", "dp_first_day")
    add_continuous("Day-0 MAP (mmHg), median [IQR]", "map_first_day")
    add_continuous("Day-0 lactate (mmol/L), median [IQR]", "lactate_first_day")
    add_continuous("Day-0 MP (J/min), median [IQR]", "mp_first_day")
    add_continuous("Mean MP over 28d (J/min), median [IQR]", "mp_mean_28d")
    add_continuous("ICU LOS (days), median [IQR]", "los")
    add_binary_pct("28-day mortality, n (%)", "mortality_28d")

    headers = ["Variable", "Overall"] + [
        f"Mild (P/F 200-300)", f"Moderate (P/F 100-200)", f"Severe (P/F <100)"
    ] + ["p-value"]
    out_lines = [
        "# Table 1. Patient characteristics by ARDS severity at day 0, MIMIC-IV ARDS cohort (28-day endpoint)",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for label, vals, p in rows:
        out_lines.append("| " + " | ".join([label] + vals + [p]) + " |")

    out_lines += [
        "",
        f"_Source: `ards_v31_v4.csv` (MIMIC-IV v3.1, first-observed-day Berlin cohort + first-day SOFA, "
        f"n = {len(stay):,} stays / {stay['subject_id'].nunique():,} unique patients). "
        f"28-day mortality defined as any death event during day 0-27 from first observed Berlin day. "
        f"ARDS severity per Berlin definition at day 0. p-values: Kruskal-Wallis test (continuous), "
        f"chi-square test (categorical)._",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}\n")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
