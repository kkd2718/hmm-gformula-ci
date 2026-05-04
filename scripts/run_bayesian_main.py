"""Bayesian main analysis: Xu + K=1 + K=5 FRE-NICE on full cohort.

Supports phase-based execution for pipelining (overlap dose_response CPU work
with the next method's NUTS GPU fit):

    --phase fit   : run NUTS only, save posterior + state, exit
    --phase dose  : load posterior + state, run dose_response, save risks
    --phase full  : (default) fit then dose_response inline

Outputs (to <out_dir>):
    {prefix}_state.npz      — posterior + benchmark state (after fit)
    {prefix}_risks.npz      — dose-response output (after dose)
    bayesian_table2.md      — combined markdown (after all dose phases)
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


# ----------------------------------------------------------------------
# State persistence helpers
# ----------------------------------------------------------------------
def _save_state_xu(bench: XuGLMMBayesian, prefix: str, out_dir: Path) -> None:
    np.savez(
        out_dir / f"{prefix}_state.npz",
        beta=bench._posterior["beta"],
        sigma_b=bench._posterior["sigma_b"],
        n_groups=bench._n_groups,
    )


def _load_state_xu(bench: XuGLMMBayesian, cohort, prefix: str, out_dir: Path) -> None:
    z = np.load(out_dir / f"{prefix}_state.npz")
    bench._posterior = {
        "beta": z["beta"],
        "sigma_b": z["sigma_b"],
    }
    bench._n_groups = int(z["n_groups"])


def _save_state_fre(bench: FRENICEBayesianBenchmark, prefix: str, out_dir: Path) -> None:
    arr = {
        "beta": bench._posterior["beta"],
        "L_chol": bench._posterior["L_chol"],
        "tau": bench._posterior["tau"],
        "B_basis": bench._B_basis,
        "sd_L": np.array(bench._sd_L),
        "n_bins": bench._n_bins, "n_dyn": bench._n_dyn,
        "n_static": bench._n_static, "t_max": bench._t_max,
        "n_groups": bench._n_groups,
        "L_has_RE_col": int(bench._L_has_RE_col),
    }
    arr["beta_L"] = np.stack(bench._beta_L) if bench._beta_L else np.zeros((0, 0))
    if bench._b_hat is not None:
        arr["b_hat"] = bench._b_hat
    np.savez(out_dir / f"{prefix}_state.npz", **arr)


def _load_state_fre(
    bench: FRENICEBayesianBenchmark, cohort, prefix: str, out_dir: Path,
) -> None:
    z = np.load(out_dir / f"{prefix}_state.npz")
    bench._posterior = {
        "beta": z["beta"],
        "L_chol": z["L_chol"],
        "tau": z["tau"],
    }
    bench._B_basis = z["B_basis"]
    bench._beta_L = [z["beta_L"][j] for j in range(z["beta_L"].shape[0])]
    bench._sd_L = list(z["sd_L"])
    bench._n_bins = int(z["n_bins"])
    bench._n_dyn = int(z["n_dyn"])
    bench._n_static = int(z["n_static"])
    bench._t_max = int(z["t_max"])
    bench._n_groups = int(z["n_groups"])
    if "L_has_RE_col" in z.files:
        bench._L_has_RE_col = bool(int(z["L_has_RE_col"]))
    if "b_hat" in z.files:
        bench._b_hat = z["b_hat"]


def _save_risks(result, prefix: str, out_dir: Path) -> None:
    np.savez(
        out_dir / f"{prefix}_risks.npz",
        bin_centers_J_min=np.array(result.bin_centers_J_min),
        bins=np.array(result.bins),
        risk_mean=result.risk_mean,
        risk_ci_low=result.risk_ci_low,
        risk_ci_high=result.risk_ci_high,
        risk_raw=result.risk_raw,
    )


def _md_table(centers, ref_bin, results: dict[str, "Tuple"]) -> list[str]:
    methods = list(results.keys())
    md = [
        "# Bayesian 4-method comparison (Table 2 surface)",
        "",
        f"_Reference bin: {ref_bin} (≈ {centers[ref_bin]:.1f} J/min). "
        "Posterior 95% credible intervals._",
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


# ----------------------------------------------------------------------
# Per-method runners (each handles fit / dose / full)
# ----------------------------------------------------------------------
def run_xu(cohort, target_bins, args, out_dir: Path):
    cfg = XuBayesianConfig(
        inference=args.inference,
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, target_accept=args.target_accept,
        svi_steps=args.svi_steps, svi_lr=args.svi_lr,
        svi_n_posterior_draws=args.svi_posterior_draws,
        n_b_draws=args.n_b_draws, n_posterior_subset=args.n_posterior_subset,
        seed=args.seed,
    )
    bench = XuGLMMBayesian(cfg)
    if args.phase in ("fit", "full"):
        print("\n=== Xu Bayesian — FIT ===")
        t0 = time.time()
        bench.fit(cohort)
        print(f"  fit time: {(time.time()-t0)/60:.1f} min")
        _save_state_xu(bench, "xu_bayesian", out_dir)
    if args.phase in ("dose", "full"):
        print("\n=== Xu Bayesian — DOSE ===")
        if args.phase == "dose":
            _load_state_xu(bench, cohort, "xu_bayesian", out_dir)
        t0 = time.time()
        result = bench.dose_response(cohort, target_bins=target_bins, refit=False)
        print(f"  dose time: {(time.time()-t0)/60:.1f} min")
        _save_risks(result, "xu_bayesian", out_dir)
        return result
    return None


def run_fre_nice(knots, prefix, cohort, target_bins, args, out_dir: Path,
                 seed_offset: int = 1):
    cfg = FRENICEBayesianConfig(
        knots=knots, inference=args.inference,
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, target_accept=args.target_accept,
        svi_steps=args.svi_steps, svi_lr=args.svi_lr,
        svi_n_posterior_draws=args.svi_posterior_draws,
        n_posterior_subset=args.n_posterior_subset,
        share_RE_on_L=args.share_RE_on_L,
        n_b_draws_per_post=5, seed=args.seed + seed_offset,
    )
    bench = FRENICEBayesianBenchmark(cfg)
    if args.phase in ("fit", "full"):
        print(f"\n=== {prefix} — FIT ===")
        t0 = time.time()
        bench.fit(cohort)
        print(f"  fit time: {(time.time()-t0)/60:.1f} min")
        _save_state_fre(bench, prefix, out_dir)
    if args.phase in ("dose", "full"):
        print(f"\n=== {prefix} — DOSE ===")
        if args.phase == "dose":
            _load_state_fre(bench, cohort, prefix, out_dir)
        t0 = time.time()
        result = bench.dose_response(cohort, target_bins=target_bins, refit=False)
        print(f"  dose time: {(time.time()-t0)/60:.1f} min")
        _save_risks(result, prefix, out_dir)
        return result
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--inference", choices=["nuts", "svi"], default="nuts")
    parser.add_argument("--phase", choices=["fit", "dose", "full"], default="full")
    parser.add_argument("--n-warmup", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--svi-steps", type=int, default=8000)
    parser.add_argument("--svi-lr", type=float, default=5e-3)
    parser.add_argument("--svi-posterior-draws", type=int, default=2000)
    parser.add_argument("--n-b-draws", type=int, default=50)
    parser.add_argument("--n-posterior-subset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-mp", type=float, default=17.0)
    parser.add_argument("--methods", nargs="+",
                        default=["xu", "K1", "K5"],
                        help="Subset of methods to run")
    parser.add_argument("--share-RE-on-L", action="store_true",
                        help="Spec ②: share FRE between Y and L equations "
                             "(refit L with extra column lambda_j * b^T B(t))")
    parser.add_argument("--K5-knots", nargs="+", type=float,
                        default=[0.0, 3.0, 7.0, 14.0, 21.0],
                        help="Knot positions for K=5 spec (sensitivity analysis)")
    parser.add_argument("--exclude-tv", nargs="*", default=[],
                        help="TV covariates to exclude (LOCO sensitivity)")
    parser.add_argument("--exclude-static", nargs="*", default=[],
                        help="Static covariates to exclude (LOCO sensitivity)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins,
        exclude_tv_cols=tuple(args.exclude_tv),
        exclude_static_cols=tuple(args.exclude_static),
    ))
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
          f"G={len(np.unique(cohort.subject_ids))} subjects. "
          f"ref_bin={ref_bin}, phase={args.phase}")

    results = {}
    if "xu" in args.methods:
        r = run_xu(cohort, target_bins, args, args.out_dir)
        if r is not None:
            results["Xu Bayesian"] = r
    if "K1" in args.methods:
        r = run_fre_nice(
            (14.0,), "fre_nice_K1", cohort, target_bins, args, args.out_dir,
            seed_offset=1,
        )
        if r is not None:
            results["FRE-NICE K=1"] = r
    if "K5" in args.methods:
        r = run_fre_nice(
            tuple(args.K5_knots), "fre_nice_K5", cohort,
            target_bins, args, args.out_dir, seed_offset=2,
        )
        if r is not None:
            results["FRE-NICE K=5"] = r
    if "K4" in args.methods:
        r = run_fre_nice(
            (0.0, 7.0, 14.0, 21.0), "fre_nice_K4", cohort,
            target_bins, args, args.out_dir, seed_offset=3,
        )
        if r is not None:
            results["FRE-NICE K=4"] = r
    if "K6" in args.methods:
        r = run_fre_nice(
            (0.0, 3.0, 7.0, 14.0, 21.0, 27.0), "fre_nice_K6", cohort,
            target_bins, args, args.out_dir, seed_offset=4,
        )
        if r is not None:
            results["FRE-NICE K=6"] = r

    if results and args.phase != "fit":
        md = _md_table(centers, ref_bin, results)
        (args.out_dir / "bayesian_table2.md").write_text(
            "\n".join(md), encoding="utf-8",
        )
        print(f"\nWrote {args.out_dir / 'bayesian_table2.md'}")


if __name__ == "__main__":
    main()
