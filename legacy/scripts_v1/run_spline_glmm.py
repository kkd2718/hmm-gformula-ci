"""Run Spline-RE NICE g-formula benchmark (K=1 scalar + K=5 spline).

Produces dose-response per MP bin for the two new methods (method 3 & 4 in
the 4-method ladder). Combine with existing Standard / Xu results for the
final 4-way Table 2.

Outputs:
    <out_dir>/spline_glmm_K1_risks.npz
    <out_dir>/spline_glmm_K5_risks.npz
    <out_dir>/spline_glmm_summary.md
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
from src.benchmarks import SplineGLMMNICEBenchmark, SplineGLMMNICEConfig
from src.training.laplace_em import LaplaceTrainingConfig


def _save_risks_npz(result, out_path: Path, prefix: str) -> None:
    np.savez(
        out_path,
        **{
            f"{prefix}__bin_centers_J_min": np.array(result.bin_centers_J_min),
            f"{prefix}__bins": np.array(result.bins),
            f"{prefix}__risk_mean": result.risk_mean,
            f"{prefix}__risk_ci_low": result.risk_ci_low,
            f"{prefix}__risk_ci_high": result.risk_ci_high,
            f"{prefix}__risk_raw": result.risk_raw,
        },
    )


def _format_md_table(
    bins, centers, mean, lo, hi, ref_bin: int, method_label: str,
) -> list[str]:
    md = [
        f"### {method_label}",
        "",
        "| MP bin | Center (J/min) | Risk % (95% CI) | RD vs ref % (95% CI) |",
        "|---|---|---|---|",
    ]
    raw_full = None  # paired RD requires raw bootstrap
    for ki, k in enumerate(bins):
        c = centers[k] if k < len(centers) else float("nan")
        risk_str = f"{100*mean[ki]:.1f} ({100*lo[ki]:.1f}–{100*hi[ki]:.1f})"
        if k == ref_bin:
            rd_str = "ref"
        else:
            rd_str = "—"
        md.append(f"| {k} | {c:.1f} | {risk_str} | {rd_str} |")
    md.append("")
    return md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--n-b-draws", type=int, default=200)
    parser.add_argument("--n-outer-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--inner-max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+", default=["K1", "K5"],
                        help="Which RE specs to run: K1 (scalar), K5 (spline).")
    parser.add_argument("--reference-mp", type=float, default=17.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = ARDSConfig(csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t)
    cohort = load_ards_cohort(base_cfg)

    # Reference bin
    edges = cohort.bin_edges_mp
    centers = np.array([
        np.sqrt(edges[k] * edges[k + 1])
        if (np.isfinite(edges[k]) and np.isfinite(edges[k + 1]) and edges[k] > 0)
        else np.nan
        for k in range(len(edges) - 1)
    ])
    ref_bin = int(np.nanargmin(np.abs(centers - args.reference_mp)))
    target_bins = list(range(args.n_bins))

    md = [
        "# Spline-RE NICE g-formula — 4-method ladder",
        "",
        f"_Reference bin: {ref_bin} (≈ {centers[ref_bin]:.1f} J/min). "
        f"Bootstrap B = {args.n_bootstrap} (refit), MC b-draws = {args.n_b_draws}._",
        "",
    ]

    training_cfg = LaplaceTrainingConfig(
        n_outer_epochs=args.n_outer_epochs,
        learning_rate=args.lr,
        inner_max_iter=args.inner_max_iter,
        print_every=20,
    )

    if "K1" in args.variants:
        print("\n=== K=1 nested case (sanity check vs Xu 2024) ===")
        cfg_k1 = SplineGLMMNICEConfig(
            knots=(14.0,),                # single knot -> constant basis (K=1)
            n_b_draws=args.n_b_draws,
            training=training_cfg,
        )
        bench_k1 = SplineGLMMNICEBenchmark(cfg_k1)
        result_k1 = bench_k1.dose_response(
            cohort, target_bins=target_bins,
            n_bootstrap=args.n_bootstrap, seed=args.seed, refit=True,
            checkpoint_path=args.out_dir / "checkpoint_K1.npz",
        )
        _save_risks_npz(
            result_k1, args.out_dir / "spline_glmm_K1_risks.npz", "spline_glmm_K1",
        )
        md.extend(_format_md_table(
            result_k1.bins, centers, result_k1.risk_mean,
            result_k1.risk_ci_low, result_k1.risk_ci_high,
            ref_bin, "K=1 nested case (scalar RE in NICE)",
        ))

    if "K5" in args.variants:
        print("\n=== Spline-RE NICE g-formula (K=5; primary proposed method) ===")
        cfg_k5 = SplineGLMMNICEConfig(
            knots=(0.0, 3.0, 7.0, 14.0, 21.0),
            n_b_draws=args.n_b_draws,
            training=training_cfg,
        )
        bench_k5 = SplineGLMMNICEBenchmark(cfg_k5)
        result_k5 = bench_k5.dose_response(
            cohort, target_bins=target_bins,
            n_bootstrap=args.n_bootstrap, seed=args.seed + 1, refit=True,
            checkpoint_path=args.out_dir / "checkpoint_K5.npz",
        )
        _save_risks_npz(
            result_k5, args.out_dir / "spline_glmm_K5_risks.npz", "spline_glmm_K5",
        )
        md.extend(_format_md_table(
            result_k5.bins, centers, result_k5.risk_mean,
            result_k5.risk_ci_low, result_k5.risk_ci_high,
            ref_bin, "Spline-RE NICE (K=5, knots=[0,3,7,14,21]) — primary",
        ))

    md_path = args.out_dir / "spline_glmm_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
