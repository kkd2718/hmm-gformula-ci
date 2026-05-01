"""Bootstrap CI for natural-course simulation across 3 methods.

For each method, B cluster bootstrap replicates (resample patients on
subject_id). In each replicate: refit the model on the resampled cohort,
then forward-simulate with each subject's OBSERVED A_t trajectory (no
intervention). Reports mean + 95% percentile CI of the natural-course
28-day mortality.

PASS criterion: cohort raw rate (25.61%) lies within the 95% CI.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks import StandardGFormula, XuGLMM, VEMSSMBenchmark, VEMConfig
from src.benchmarks._resample import cluster_bootstrap_indices, slice_cohort
from src.training.variational_em import TrainingConfig
from scripts.sanity_check import (
    natural_course_simulation, standard_natural_course, xu_natural_course,
    bin_stratified_raw_mortality,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--vem-epochs", type=int, default=50)
    parser.add_argument("--vem-z-lag-treatment", action="store_true",
                        help="Option B (default for primary natural-course).")
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] cohort ...")
    cohort = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t,
    ))

    bin_df, overall = bin_stratified_raw_mortality(
        args.csv, n_bins=args.n_bins, max_t=args.max_t,
    )
    print(f"[ref] cohort overall 28-day mortality = {overall * 100:.2f}%")

    rng = np.random.default_rng(args.seed)
    out: dict[str, list[float]] = {"standard": [], "xu": [], "vem": []}

    for b in range(args.n_bootstrap):
        idx = cluster_bootstrap_indices(cohort.subject_ids, rng)
        boot_cohort = slice_cohort(cohort, idx)

        # Standard
        m_std = StandardGFormula()
        m_std.fit(boot_cohort)
        out["standard"].append(standard_natural_course(m_std, boot_cohort))

        # Xu
        m_xu = XuGLMM()
        m_xu.fit(boot_cohort)
        out["xu"].append(xu_natural_course(m_xu, boot_cohort, n_b_draws=50))

        # VEM
        m_vem = VEMSSMBenchmark(VEMConfig(
            training=TrainingConfig(
                n_epochs=args.vem_epochs, learning_rate=1e-2, n_mc_samples=4,
                smoothness_lambda=0.02,
            ),
            z_depends_on_treatment_lag=args.vem_z_lag_treatment,
        ))
        m_vem.fit(boot_cohort)
        out["vem"].append(natural_course_simulation(
            m_vem._model, boot_cohort, n_mc_subjects=boot_cohort.Y.shape[0],
        ))

        if (b + 1) % 5 == 0:
            print(f"  rep {b + 1}/{args.n_bootstrap}: "
                  f"std={out['standard'][-1]*100:.2f} "
                  f"xu={out['xu'][-1]*100:.2f} "
                  f"vem={out['vem'][-1]*100:.2f}")

    arr = {k: np.array(v) for k, v in out.items()}
    np.savez(args.out_dir / "natural_course_bootstrap.npz",
             cohort_raw=overall, **arr)

    md = [
        "# Appendix F: Natural-course simulation with bootstrap CI",
        "",
        f"_Cluster bootstrap on subject_id, B = {args.n_bootstrap} refit replicates. "
        f"Each replicate: resample patients, refit model, forward-simulate with "
        f"observed A_t trajectory (no intervention). Cohort raw 28-day mortality = "
        f"**{overall * 100:.2f}%**._",
        "",
        "| Method | Mean (%) | 95% CI (%) | Includes cohort 25.61%? |",
        "|---|---|---|---|",
    ]
    for key, label in [
        ("standard", "Standard parametric g-formula"),
        ("xu", "Xu 2024 GLMM g-computation"),
        ("vem", "VEM-SSM g-formula (proposed)"),
    ]:
        a = arr[key] * 100
        m, lo, hi = a.mean(), np.quantile(a, 0.025), np.quantile(a, 0.975)
        includes = "YES" if (lo <= overall * 100 <= hi) else "NO"
        md.append(f"| {label} | {m:.2f} | ({lo:.2f}, {hi:.2f}) | {includes} |")

    md.extend([
        "",
        "## Interpretation",
        "",
        "Natural-course simulation under each method's fitted generative model "
        "is a calibration check (Taubman 2009 *IJE*; Keil 2014 *Epidemiology*; "
        "McGrath 2020 *Patterns*): a method whose 95% CI contains the cohort raw "
        "rate is well-calibrated to the observed data distribution. This does NOT "
        "validate counterfactual causal estimates — only confirms that the "
        "generative model can re-produce the observed factual distribution.",
    ])
    md_path = args.out_dir / "appendix_f_natural_course.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
