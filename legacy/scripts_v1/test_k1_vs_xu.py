"""K=1 SplineGLMMNICE vs Xu 2024 GLMM — convergence sanity check.

Both methods use a scalar random intercept on outcome. They differ in:
- Loss function (Laplace marginal LL via PyTorch autograd vs Xu's joint Newton + MoM σ_b)
- Counterfactual procedure (forward L sim vs observed L plug-in)
- Optimizer (Adam vs Newton-Raphson)

Under the convergence conditions where these implementation differences are
small (large N, well-fit GLMM, similar L distribution between observed and
forward-simulated), the two methods' dose-response should agree within MC
error. Large divergence flags a bug or a meaningful framework gap.

Run on a single full-cohort fit (no bootstrap) for both, compare per-bin
risks and report max absolute difference + correlation.
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
from src.benchmarks import (
    XuGLMM, SplineGLMMNICEBenchmark, SplineGLMMNICEConfig,
)
from src.training.laplace_em import LaplaceTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--n-b-draws", type=int, default=200)
    parser.add_argument("--n-outer-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t,
    ))
    target_bins = list(range(args.n_bins))

    print("=== Fitting Xu 2024 GLMM (scalar b_i, MSM-style) ===")
    xu = XuGLMM(n_b_draws=args.n_b_draws)
    xu.fit(cohort)
    print(f"  fitted: sigma_b = {xu._sigma_b:.4f}")
    xu_result = xu.dose_response(
        cohort, target_bins=target_bins,
        n_bootstrap=1, seed=args.seed, refit=False,
    )

    print("\n=== Fitting K=1 SplineGLMMNICE (scalar b_i in NICE framework) ===")
    cfg = SplineGLMMNICEConfig(
        knots=(14.0,),
        n_b_draws=args.n_b_draws,
        training=LaplaceTrainingConfig(
            n_outer_epochs=args.n_outer_epochs, learning_rate=5e-3,
            print_every=20,
        ),
    )
    k1 = SplineGLMMNICEBenchmark(cfg)
    k1.fit(cohort)
    sig_b = float(np.sqrt(k1._L_chol_np[0, 0] ** 2))
    print(f"  fitted: sigma_b (normalized basis) = {sig_b:.4f}")
    k1_result = k1.dose_response(
        cohort, target_bins=target_bins,
        n_bootstrap=1, seed=args.seed + 1, refit=False,
    )

    diff = k1_result.risk_mean - xu_result.risk_mean
    corr = float(np.corrcoef(k1_result.risk_mean, xu_result.risk_mean)[0, 1])
    print("\n=== Comparison ===")
    print(f"  per-bin abs diff: max={np.abs(diff).max() * 100:.2f}%p, "
          f"mean={np.abs(diff).mean() * 100:.2f}%p")
    print(f"  Pearson correlation across bins: {corr:.4f}")

    md = [
        "# K=1 SplineGLMMNICE vs Xu 2024 — convergence sanity check",
        "",
        f"_Both methods fit on full cohort (N={cohort.Y.shape[0]}). "
        f"n_b_draws = {args.n_b_draws}. No bootstrap (point estimate)._",
        "",
        "| MP bin | Center (J/min) | Xu risk % | K=1 NICE risk % | Diff (%p) |",
        "|---|---|---|---|---|",
    ]
    for ki, k in enumerate(target_bins):
        c = xu_result.bin_centers_J_min[k] if k < len(xu_result.bin_centers_J_min) else float("nan")
        md.append(
            f"| {k} | {c:.1f} | {100*xu_result.risk_mean[ki]:.1f} | "
            f"{100*k1_result.risk_mean[ki]:.1f} | {100*diff[ki]:+.2f} |"
        )
    md.extend([
        "",
        f"**Max abs diff:** {np.abs(diff).max() * 100:.2f}%p",
        f"**Mean abs diff:** {np.abs(diff).mean() * 100:.2f}%p",
        f"**Pearson correlation across bins:** {corr:.4f}",
        "",
        "## Interpretation",
        "",
        "Large divergence (>5%p any bin OR corr < 0.95) suggests:",
        "- (a) Implementation bug in either Xu or K=1 NICE.",
        "- (b) Framework gap (NICE forward L sim vs Xu's observed L plug-in) "
        "is empirically large for this cohort — meaningful finding for the manuscript.",
        "",
        "Small divergence confirms the two implementations behave equivalently "
        "in the scalar-RE limit, validating the K=1 reduction claim.",
    ])
    md_path = args.out_dir / "k1_vs_xu_comparison.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {md_path}")
    np.savez(
        args.out_dir / "k1_vs_xu.npz",
        bins=np.array(target_bins),
        xu_risk=xu_result.risk_mean,
        k1_risk=k1_result.risk_mean,
        diff=diff,
        corr=corr,
    )


if __name__ == "__main__":
    main()
