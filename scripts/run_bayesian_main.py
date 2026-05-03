"""Bayesian main analysis: Xu + K=1 + K=5 FRE-NICE on full cohort.

Sequentially fits the three Bayesian methods, saves posterior samples and
counterfactual dose-response (posterior mean + 95% credible interval) per
target MP bin.

Outputs (to <out_dir>):
    xu_bayesian_risks.npz
    fre_nice_K1_risks.npz
    fre_nice_K5_risks.npz
    posterior_xu.npz, posterior_K1.npz, posterior_K5.npz
    bayesian_table2.md       — combined dose-response markdown
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
from src.benchmarks import (
    XuGLMMBayesian, XuBayesianConfig,
    FRENICEBayesianBenchmark, FRENICEBayesianConfig,
)


def _save(result, posterior, prefix: str, out_dir: Path) -> None:
    np.savez(
        out_dir / f"{prefix}_risks.npz",
        bin_centers_J_min=np.array(result.bin_centers_J_min),
        bins=np.array(result.bins),
        risk_mean=result.risk_mean,
        risk_ci_low=result.risk_ci_low,
        risk_ci_high=result.risk_ci_high,
        risk_raw=result.risk_raw,
    )
    np.savez(out_dir / f"posterior_{prefix}.npz", **posterior)


def _md_table(centers, ref_bin, results: dict[str, "Tuple"]) -> list[str]:
    """Build a side-by-side dose-response table across methods."""
    methods = list(results.keys())
    md = [
        "# Bayesian 4-method comparison (Table 2 surface)",
        "",
        f"_Reference bin: {ref_bin} (≈ {centers[ref_bin]:.1f} J/min). "
        "Posterior 95% credible intervals; no bootstrap (Bayesian standard)._",
        "",
    ]
    headers = ["MP bin", "Center (J/min)"]
    for m in methods:
        headers.extend([f"{m} risk %", f"{m} 95% CI"])
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    n_bins = len(results[methods[0]].risk_mean)
    for k in range(n_bins):
        c = centers[k] if k < len(centers) else float("nan")
        row = [str(k), f"{c:.1f}"]
        for m in methods:
            r = results[m]
            row.append(f"{100*r.risk_mean[k]:.1f}")
            row.append(f"({100*r.risk_ci_low[k]:.1f}–{100*r.risk_ci_high[k]:.1f})")
        md.append("| " + " | ".join(row) + " |")
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--n-warmup", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--n-b-draws", type=int, default=50)
    parser.add_argument("--n-posterior-subset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-mp", type=float, default=17.0)
    parser.add_argument("--methods", nargs="+",
                        default=["xu", "K1", "K5"],
                        help="Subset of methods to run")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_ards_cohort(ARDSConfig(csv_path=args.csv, n_bins=args.n_bins))
    target_bins = list(range(args.n_bins))
    edges = cohort.bin_edges_mp
    centers = np.array([
        np.sqrt(edges[k] * edges[k + 1])
        if (np.isfinite(edges[k]) and np.isfinite(edges[k + 1]) and edges[k] > 0)
        else float("nan")
        for k in range(len(edges) - 1)
    ])
    ref_bin = int(np.nanargmin(np.abs(centers - args.reference_mp)))
    print(f"Cohort: N={cohort.Y.shape[0]} stays, "
          f"G={len(np.unique(cohort.subject_ids))} subjects. ref_bin={ref_bin}")

    results = {}

    if "xu" in args.methods:
        print("\n=== Method 2: Xu Bayesian GLMM (scalar RE, MSM) ===")
        cfg = XuBayesianConfig(
            n_warmup=args.n_warmup, n_samples=args.n_samples,
            n_chains=args.n_chains, target_accept=args.target_accept,
            n_b_draws=args.n_b_draws, n_posterior_subset=args.n_posterior_subset,
            seed=args.seed,
        )
        bench = XuGLMMBayesian(cfg)
        t0 = time.time()
        bench.fit(cohort)
        print(f"  fit time: {(time.time()-t0)/60:.1f} min")
        result = bench.dose_response(cohort, target_bins=target_bins, refit=False)
        _save(result, bench._posterior, "xu_bayesian", args.out_dir)
        results["Xu Bayesian"] = result

    if "K1" in args.methods:
        print("\n=== Method 3: FRE-NICE Bayesian K=1 (scalar RE, NICE) ===")
        cfg = FRENICEBayesianConfig(
            knots=(14.0,),
            n_warmup=args.n_warmup, n_samples=args.n_samples,
            n_chains=args.n_chains, target_accept=args.target_accept,
            n_posterior_subset=args.n_posterior_subset,
            n_b_draws_per_post=5, seed=args.seed + 1,
        )
        bench = FRENICEBayesianBenchmark(cfg)
        t0 = time.time()
        bench.fit(cohort)
        print(f"  fit time: {(time.time()-t0)/60:.1f} min")
        result = bench.dose_response(cohort, target_bins=target_bins, refit=False)
        _save(result, bench._posterior, "fre_nice_K1", args.out_dir)
        results["FRE-NICE K=1"] = result

    if "K5" in args.methods:
        print("\n=== Method 4: FRE-NICE Bayesian K=5 (functional RE, NICE) ===")
        cfg = FRENICEBayesianConfig(
            knots=(0.0, 3.0, 7.0, 14.0, 21.0),
            n_warmup=args.n_warmup, n_samples=args.n_samples,
            n_chains=args.n_chains, target_accept=args.target_accept,
            n_posterior_subset=args.n_posterior_subset,
            n_b_draws_per_post=5, seed=args.seed + 2,
        )
        bench = FRENICEBayesianBenchmark(cfg)
        t0 = time.time()
        bench.fit(cohort)
        print(f"  fit time: {(time.time()-t0)/60:.1f} min")
        result = bench.dose_response(cohort, target_bins=target_bins, refit=False)
        _save(result, bench._posterior, "fre_nice_K5", args.out_dir)
        results["FRE-NICE K=5"] = result

    # Combined markdown
    if results:
        md = _md_table(centers, ref_bin, results)
        (args.out_dir / "bayesian_table2.md").write_text(
            "\n".join(md), encoding="utf-8",
        )
        print(f"\nWrote {args.out_dir / 'bayesian_table2.md'}")


if __name__ == "__main__":
    main()
