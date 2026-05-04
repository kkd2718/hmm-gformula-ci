"""GPU dose_response runner — loads existing posterior state, runs JAX dose.

Outputs (overwrites existing risks if present):
    {prefix}_risks.npz
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
from src.benchmarks.dose_response_jax import fre_nice_dose_response_jax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=str,
                        choices=["fre_nice_K1", "fre_nice_K5"])
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--ref-bin", type=int, default=16)
    parser.add_argument("--n-posterior-subset", type=int, default=200)
    parser.add_argument("--n-b-draws-per-post", type=int, default=5)
    parser.add_argument("--share-RE-on-L", action="store_true",
                        help="Spec ②: forward L sim uses augmented beta_L")
    parser.add_argument("--exclude-tv", nargs="*", default=[])
    parser.add_argument("--exclude-static", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== JAX dose_response: {args.prefix} ===")
    cohort = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins,
        exclude_tv_cols=tuple(args.exclude_tv),
        exclude_static_cols=tuple(args.exclude_static),
    ))
    L_obs = cohort.L_dyn.numpy().astype(np.float32)
    C_static = cohort.C_static.numpy().astype(np.float32)
    K_A = cohort.feature_layout["n_bins"]
    target_bins = list(range(args.n_bins))
    edges = cohort.bin_edges_mp
    centers = np.array([
        np.sqrt(edges[k] * edges[k + 1])
        if (np.isfinite(edges[k]) and np.isfinite(edges[k + 1]) and edges[k] > 0)
        else float("nan") for k in range(len(edges) - 1)
    ])

    state = np.load(args.state_dir / f"{args.prefix}_state.npz")
    posterior = {
        "beta": state["beta"],
        "L_chol": state["L_chol"],
        "tau": state["tau"],
    }
    B_basis = state["B_basis"]
    beta_L_list = [state["beta_L"][j] for j in range(state["beta_L"].shape[0])]
    sd_L_list = list(state["sd_L"])
    print(f"  state loaded: posterior beta shape {posterior['beta'].shape}, "
          f"L_chol {posterior['L_chol'].shape}, B_basis {B_basis.shape}")

    t0 = time.time()
    risk_mat = fre_nice_dose_response_jax(
        posterior=posterior,
        B_basis=B_basis,
        beta_L_list=beta_L_list,
        sd_L_list=sd_L_list,
        L_obs=L_obs,
        C_static=C_static,
        target_bins=target_bins,
        K_A=K_A, ref_bin=args.ref_bin,
        n_posterior_subset=args.n_posterior_subset,
        n_b_draws_per_post=args.n_b_draws_per_post,
        seed=args.seed,
        share_RE_on_L=args.share_RE_on_L,
    )
    print(f"\n  total time: {(time.time()-t0)/60:.1f} min")
    print(f"  risk_mat shape: {risk_mat.shape} (S, n_bins)")

    risk_mean = risk_mat.mean(axis=0)
    risk_ci_low = np.quantile(risk_mat, 0.025, axis=0)
    risk_ci_high = np.quantile(risk_mat, 0.975, axis=0)

    out_path = args.out_dir / f"{args.prefix}_risks.npz"
    np.savez(
        out_path,
        bin_centers_J_min=centers,
        bins=np.array(target_bins),
        risk_mean=risk_mean,
        risk_ci_low=risk_ci_low,
        risk_ci_high=risk_ci_high,
        risk_raw=risk_mat.T,
    )
    print(f"  wrote {out_path}")
    print(f"  bin 16 (ref): mean={100*risk_mean[16]:.1f}% "
          f"({100*risk_ci_low[16]:.1f}-{100*risk_ci_high[16]:.1f})")
    print(f"  bin 17:       mean={100*risk_mean[17]:.1f}% "
          f"({100*risk_ci_low[17]:.1f}-{100*risk_ci_high[17]:.1f})")


if __name__ == "__main__":
    main()
