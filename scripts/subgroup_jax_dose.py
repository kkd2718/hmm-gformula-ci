"""Subgroup analysis: K=5 FRE-NICE dose on cohort subgroups.

Uses existing fitted posterior (K=5 state.npz). For each subgroup defined by
static covariate condition, subsets cohort and runs JAX dose_response.
Reports per-subgroup risk at low MP (bin 7, ≈3 J/min) vs high MP (bin 16, ≈17 J/min).

Subgroups defined by Berlin severity (mild/moderate/severe), age (>=65), BMI (>=30),
PEEP threshold (<8 vs >=8 at day 0).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks._resample import slice_cohort
from src.benchmarks.dose_response_jax import fre_nice_dose_response_jax


def define_subgroups(cohort, csv_path: Path) -> dict:
    """Return dict of subgroup_name -> bool mask (N,) using clinical
    reference cutoffs. C_static in cohort is z-scored, so original values
    are reloaded from CSV and aligned to cohort stay_id order.

    Reference cutoffs:
      - Age >=65: WHO/general clinical older-adult definition
      - BMI >=30: WHO Class I obesity (non-Asian populations); the
        present cohort is MIMIC-IV (predominantly non-Asian)
      - Charlson >=5: high comorbidity burden (Charlson 1987; multiple
        validation studies threshold at 3-5; we use 5 to capture the
        upper tail with greater clinical homogeneity)
    """
    import pandas as pd
    sev = cohort.severity_label
    df = pd.read_csv(csv_path)
    by_stay = df.sort_values(["stay_id", "day_num"]).groupby("stay_id").first().reset_index()
    by_stay = by_stay.set_index("stay_id")
    # Align to cohort.subject_ids stay ordering — assumes cohort.subject_ids
    # is keyed by stay_id (as built in ards.py). If hash collision, use the
    # cohort's internal stay order via reset() utility.
    if hasattr(cohort, "stay_ids"):
        order = cohort.stay_ids
    else:
        # Fallback: cohort.subject_ids order matches CSV first-seen order
        order = sorted(df["stay_id"].unique())
    aligned = by_stay.loc[order]
    age = aligned["anchor_age"].to_numpy()
    bmi = aligned["bmi_imputed"].to_numpy()
    charl = aligned["charlson_index"].to_numpy()

    subs = {
        "mild": sev == "mild",
        "moderate": sev == "moderate",
        "severe": sev == "severe",
        "age_geq65": age >= 65,
        "age_lt65": age < 65,
        "obese_geq30": bmi >= 30,
        "non_obese_lt30": bmi < 30,
        "charlson_geq5": charl >= 5,
        "charlson_lt5": charl < 5,
    }
    return subs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--state-path", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--ref-bin", type=int, default=16)
    parser.add_argument("--low-bin", type=int, default=7)
    parser.add_argument("--n-posterior-subset", type=int, default=200)
    parser.add_argument("--n-b-draws-per-post", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    full = load_ards_cohort(ARDSConfig(csv_path=args.csv, n_bins=args.n_bins))
    state = np.load(args.state_path)
    posterior = {
        "beta": state["beta"], "L_chol": state["L_chol"], "tau": state["tau"],
    }
    B_basis = state["B_basis"]
    beta_L_list = [state["beta_L"][j] for j in range(state["beta_L"].shape[0])]
    sd_L_list = list(state["sd_L"])
    K_A = int(state["n_bins"])

    subgroups = define_subgroups(full, args.csv)
    md = [
        "# Subgroup analysis: K=5 FRE-NICE dose-response",
        "",
        f"_Reference bin: {args.ref_bin}, low bin: {args.low_bin}. "
        "Posterior 95% credible intervals._",
        "",
        "| Subgroup | N | Low MP risk % | Ref MP risk % | RD (high vs low) |",
        "|---|---|---|---|---|",
    ]
    target_bins = [args.low_bin, args.ref_bin, 17]   # low, ref, high

    rows = []
    for sub_name, mask in subgroups.items():
        idx = np.where(mask)[0]
        if len(idx) < 100:
            print(f"[skip] {sub_name}: N={len(idx)} too small")
            continue
        sub = slice_cohort(full, idx)
        L_obs = sub.L_dyn.numpy().astype(np.float64)
        C_static = sub.C_static.numpy().astype(np.float64)
        print(f"\n=== {sub_name}: N={sub.Y.shape[0]} ===")
        risk_mat = fre_nice_dose_response_jax(
            posterior=posterior, B_basis=B_basis,
            beta_L_list=beta_L_list, sd_L_list=sd_L_list,
            L_obs=L_obs, C_static=C_static,
            target_bins=target_bins,
            K_A=K_A, ref_bin=args.ref_bin,
            n_posterior_subset=args.n_posterior_subset,
            n_b_draws_per_post=args.n_b_draws_per_post,
            seed=args.seed,
        )
        risk_low = 100 * risk_mat[:, 0].mean()
        risk_ref = 100 * risk_mat[:, 1].mean()
        risk_high = 100 * risk_mat[:, 2].mean()
        rd = risk_high - risk_low
        rd_lo = 100 * np.quantile(risk_mat[:, 2] - risk_mat[:, 0], 0.025)
        rd_hi = 100 * np.quantile(risk_mat[:, 2] - risk_mat[:, 0], 0.975)
        md.append(f"| {sub_name} | {len(idx)} | "
                  f"{risk_low:.1f} | {risk_ref:.1f} | "
                  f"{rd:+.1f} ({rd_lo:+.1f}, {rd_hi:+.1f}) |")
        rows.append({
            "name": sub_name, "n": len(idx),
            "risk_low": risk_low, "risk_ref": risk_ref, "risk_high": risk_high,
            "rd": rd, "rd_lo": rd_lo, "rd_hi": rd_hi,
        })

    args.out_md.write_text("\n".join(md), encoding="utf-8")
    np.savez(
        args.out_md.with_suffix(".npz"),
        names=[r["name"] for r in rows],
        ns=[r["n"] for r in rows],
        risk_low=[r["risk_low"] for r in rows],
        risk_ref=[r["risk_ref"] for r in rows],
        risk_high=[r["risk_high"] for r in rows],
        rd=[r["rd"] for r in rows], rd_lo=[r["rd_lo"] for r in rows],
        rd_hi=[r["rd_hi"] for r in rows],
    )
    print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
