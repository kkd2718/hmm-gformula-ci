"""Unit test: numpy vs JAX dose_response on a small cohort subset.

Same posterior, same baseline subjects, run BOTH numpy and JAX implementations,
compare per-bin risk_mean. Tolerance: <0.5%p across all bins (Monte Carlo
noise threshold).
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
from src.benchmarks.fre_nice_bayesian import (
    FRENICEBayesianBenchmark, FRENICEBayesianConfig,
)
from src.benchmarks.dose_response_jax import fre_nice_dose_response_jax
from src.models.spline_glmm import natural_cubic_basis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--state-path", required=True, type=Path,
                        help="Path to fre_nice_K{1,5}_state.npz from full fit")
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--n-subjects", type=int, default=500,
                        help="Subset size for fast comparison")
    parser.add_argument("--n-posterior-subset", type=int, default=20)
    parser.add_argument("--n-b-draws-per-post", type=int, default=5)
    parser.add_argument("--knots", nargs="+", type=float,
                        default=[0.0, 3.0, 7.0, 14.0, 21.0])
    parser.add_argument("--ref-bin", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    full = load_ards_cohort(ARDSConfig(csv_path=args.csv))
    rng = np.random.default_rng(0)
    uniq = np.unique(full.subject_ids)
    sel = rng.choice(uniq, size=args.n_subjects, replace=False)
    mask = np.isin(full.subject_ids, sel)
    idx = np.where(mask)[0]
    cohort = slice_cohort(full, idx)
    print(f"Subset: N stays={cohort.Y.shape[0]}, "
          f"G subjects={len(np.unique(cohort.subject_ids))}")

    # Load state
    state = np.load(args.state_path)
    posterior = {
        "beta": state["beta"], "L_chol": state["L_chol"], "tau": state["tau"],
    }
    B_basis = state["B_basis"]
    beta_L_list = [state["beta_L"][j] for j in range(state["beta_L"].shape[0])]
    sd_L_list = list(state["sd_L"])
    K_A = int(state["n_bins"])

    target_bins = list(range(K_A))

    # ----- numpy implementation: rebuild bench + load posterior + run -----
    knots = tuple(args.knots)
    cfg = FRENICEBayesianConfig(
        knots=knots, n_posterior_subset=args.n_posterior_subset,
        n_b_draws_per_post=args.n_b_draws_per_post,
        ref_bin=args.ref_bin, seed=args.seed,
    )
    bench = FRENICEBayesianBenchmark(cfg)
    bench._B_basis = B_basis
    bench._beta_L = beta_L_list
    bench._sd_L = sd_L_list
    bench._n_bins = K_A
    bench._n_dyn = int(state["n_dyn"])
    bench._n_static = int(state["n_static"])
    bench._t_max = int(state["t_max"])
    bench._n_groups = int(state["n_groups"])
    bench._posterior = posterior

    print(f"\n=== numpy dose_response (subset N={args.n_subjects}) ===")
    t0 = time.time()
    np_result = bench.dose_response(
        cohort, target_bins=target_bins, refit=False,
    )
    np_time = time.time() - t0
    print(f"  numpy time: {np_time:.1f}s")

    # ----- JAX implementation: same posterior, same subset cohort -----
    L_obs = cohort.L_dyn.numpy().astype(np.float64)
    C_static = cohort.C_static.numpy().astype(np.float64)

    print(f"\n=== JAX dose_response (subset N={args.n_subjects}) ===")
    t0 = time.time()
    jax_risk_mat = fre_nice_dose_response_jax(
        posterior=posterior,
        B_basis=B_basis,
        beta_L_list=beta_L_list,
        sd_L_list=sd_L_list,
        L_obs=L_obs, C_static=C_static,
        target_bins=target_bins,
        K_A=K_A, ref_bin=args.ref_bin,
        n_posterior_subset=args.n_posterior_subset,
        n_b_draws_per_post=args.n_b_draws_per_post,
        seed=args.seed,
    )
    jax_time = time.time() - t0
    print(f"  jax time: {jax_time:.1f}s, speedup: {np_time/jax_time:.1f}x")

    jax_mean = jax_risk_mat.mean(axis=0)

    # ----- compare -----
    print("\n=== Comparison ===")
    md = [
        "# numpy vs JAX dose_response unit test",
        "",
        f"_Cohort subset: N={args.n_subjects} unique subjects, "
        f"posterior_subset={args.n_posterior_subset}, "
        f"n_b_draws_per_post={args.n_b_draws_per_post}._",
        "",
        f"_numpy time: {np_time:.1f}s, JAX time: {jax_time:.1f}s, "
        f"speedup: {np_time/jax_time:.1f}x_",
        "",
        "| Bin | MP center | numpy mean | JAX mean | Diff (%p) |",
        "|---|---|---|---|---|",
    ]
    diffs = []
    for k in target_bins:
        np_v = 100 * np_result.risk_mean[k]
        jax_v = 100 * jax_mean[k]
        d = jax_v - np_v
        diffs.append(d)
        c = np_result.bin_centers_J_min[k]
        md.append(f"| {k} | {c:.2f} | {np_v:.2f} | {jax_v:.2f} | {d:+.3f} |")
        print(f"  bin {k:2d} (MP={c:5.2f}): numpy={np_v:5.2f}, "
              f"jax={jax_v:5.2f}, diff={d:+.3f}%p")
    diffs = np.array(diffs)
    md.extend([
        "",
        f"**Max abs diff:** {np.abs(diffs).max():.3f}%p",
        f"**Mean abs diff:** {np.abs(diffs).mean():.3f}%p",
        f"**Tolerance for equivalence:** 0.5%p (MC noise threshold)",
        f"**Verdict:** {'EQUIVALENT' if np.abs(diffs).max() < 0.5 else 'DRIFT'}",
    ])
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nMax abs diff: {np.abs(diffs).max():.3f}%p")
    print(f"Mean abs diff: {np.abs(diffs).mean():.3f}%p")
    print(f"Verdict: {'EQUIVALENT' if np.abs(diffs).max() < 0.5 else 'DRIFT'}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
