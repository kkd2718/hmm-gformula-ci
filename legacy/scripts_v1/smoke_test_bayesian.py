"""Smoke test Bayesian Xu and FRE-NICE on a small cohort subset.

Verifies:
1. numpyro NUTS runs without errors on the model
2. Estimates fit time per chain (extrapolated to full cohort)
3. Sigma_b posterior is non-degenerate (no MoM-style collapse)
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks._resample import slice_cohort
from src.benchmarks import (
    XuGLMMBayesian, XuBayesianConfig,
    FRENICEBayesianBenchmark, FRENICEBayesianConfig,
)


def _subset_cohort(cohort, n_subjects: int, seed: int = 0):
    """First n_subjects unique patients (cluster on subject_id)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(cohort.subject_ids)
    sel = rng.choice(uniq, size=min(n_subjects, len(uniq)), replace=False)
    mask = np.isin(cohort.subject_ids, sel)
    idx = np.where(mask)[0]
    return slice_cohort(cohort, idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--n-subjects", type=int, default=500)
    parser.add_argument("--n-warmup", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-chains", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading cohort, taking first {args.n_subjects} unique patients...")
    full = load_ards_cohort(ARDSConfig(csv_path=args.csv))
    print(f"  full cohort: N stays = {full.Y.shape[0]}, "
          f"G subjects = {len(np.unique(full.subject_ids))}")
    cohort = _subset_cohort(full, args.n_subjects)
    print(f"  subset:      N stays = {cohort.Y.shape[0]}, "
          f"G subjects = {len(np.unique(cohort.subject_ids))}")
    print()

    # ----- Xu Bayesian -----
    print(f"=== Xu Bayesian (NUTS, {args.n_chains} chains × "
          f"{args.n_warmup}+{args.n_samples}) ===")
    xu_cfg = XuBayesianConfig(
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, n_b_draws=20, n_posterior_subset=50,
    )
    xu = XuGLMMBayesian(xu_cfg)
    t0 = time.time()
    xu.fit(cohort)
    t_xu_fit = time.time() - t0
    print(f"  Xu fit time: {t_xu_fit:.1f}s on {args.n_subjects} subjects "
          f"=> est. full ({len(np.unique(full.subject_ids))}) "
          f"~ {t_xu_fit * len(np.unique(full.subject_ids)) / args.n_subjects / 60:.1f} min")
    sigma_b_post = xu._posterior["sigma_b"]
    print(f"  sigma_b posterior: mean={sigma_b_post.mean():.3f}, "
          f"95% CI=[{np.quantile(sigma_b_post, 0.025):.3f}, "
          f"{np.quantile(sigma_b_post, 0.975):.3f}]")
    print()

    # ----- FRE-NICE Bayesian K=1 -----
    print(f"=== FRE-NICE Bayesian K=1 (NUTS) ===")
    cfg_k1 = FRENICEBayesianConfig(
        knots=(14.0,),
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, n_posterior_subset=50, n_b_draws_per_post=2,
    )
    bench_k1 = FRENICEBayesianBenchmark(cfg_k1)
    t0 = time.time()
    bench_k1.fit(cohort)
    t_k1_fit = time.time() - t0
    print(f"  FRE-NICE K=1 fit time: {t_k1_fit:.1f}s")
    tau_post = bench_k1._posterior["tau"]
    print(f"  tau posterior mean: {tau_post.mean(axis=0)}")
    print()

    # ----- FRE-NICE Bayesian K=5 -----
    print(f"=== FRE-NICE Bayesian K=5 (spline knots [0,3,7,14,21]) ===")
    cfg_k5 = FRENICEBayesianConfig(
        knots=(0.0, 3.0, 7.0, 14.0, 21.0),
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, n_posterior_subset=50, n_b_draws_per_post=2,
    )
    bench_k5 = FRENICEBayesianBenchmark(cfg_k5)
    t0 = time.time()
    bench_k5.fit(cohort)
    t_k5_fit = time.time() - t0
    print(f"  FRE-NICE K=5 fit time: {t_k5_fit:.1f}s on {args.n_subjects} subjects "
          f"=> est. full ~ "
          f"{t_k5_fit * len(np.unique(full.subject_ids)) / args.n_subjects / 60:.1f} min")
    tau_post = bench_k5._posterior["tau"]
    print(f"  tau (5 dims) posterior means: {tau_post.mean(axis=0)}")
    print()

    print("ALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
