"""Generate Table S1: simulation under known DGP × confounding-strength sweep × estimators.

DGP per src.simulation.dgp (Soohoo & Arah 2023 + AR(1) latent), with 4 gamma cells
and 500 replicates per cell. Reports bias / empirical SE / RMSE / 95% coverage of
the true marginal counterfactual 28-day risk for each MP bin.

Outputs:
    <out_dir>/table_s1_simulation.md
    <out_dir>/table_s1_raw.npz
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.dgp import simulate_tv_latent_confounding, DGPConfig
from src.data.ards import ARDSCohort
from src.benchmarks import StandardGFormula, XuGLMM, VEMSSMBenchmark, VEMConfig
from src.training.variational_em import TrainingConfig


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _sample_to_cohort(sample, K: int) -> ARDSCohort:
    """Wrap a DGPSample as an ARDSCohort for benchmark consumption."""
    N, T = sample.A.shape
    A_bin = np.zeros((N, T, K), dtype=np.float32)
    A_bin[np.arange(N)[:, None], np.arange(T)[None, :], sample.A] = 1.0
    L_dyn = sample.L[:, :, None].astype(np.float32)
    C_static = np.zeros((N, 1), dtype=np.float32)
    Y = sample.Y[:, :, None].astype(np.float32)
    at_risk = sample.at_risk[:, :, None].astype(np.float32)
    t_norm = (np.arange(T, dtype=np.float32) / max(T - 1, 1))[None, :, None]
    t_norm = np.broadcast_to(t_norm, (N, T, 1)).astype(np.float32)
    C_broadcast = np.broadcast_to(C_static[:, None, :], (N, T, 1)).astype(np.float32)
    drivers = np.concatenate([A_bin, L_dyn, C_broadcast], axis=-1)
    covariates = np.concatenate([drivers, t_norm], axis=-1)
    edges = np.linspace(0, K, K + 1)
    return ARDSCohort(
        Y=torch.from_numpy(Y),
        A_bin=torch.from_numpy(A_bin),
        L_dyn=torch.from_numpy(L_dyn),
        C_static=torch.from_numpy(C_static),
        at_risk=torch.from_numpy(at_risk),
        t_norm=torch.from_numpy(t_norm),
        drivers=torch.from_numpy(drivers),
        covariates=torch.from_numpy(covariates),
        bin_edges_mp=edges,
        stay_ids=np.arange(N),
        subject_ids=np.arange(N),
        severity_label=np.array(["moderate"] * N),
        feature_layout={
            "n_bins": K, "n_dyn": 1, "n_static": 1,
            "drivers_width": K + 2, "covariates_width": K + 3,
            "tv_cols": ["L0"], "static_cols": ["C0"],
            "mp_bin_edges": edges.tolist(),
            "log_mp_mu": 0.0, "log_mp_sd": 1.0,
            "bin_slice": (0, K),
        },
    )


def _true_psi(cfg: DGPConfig, k: int, M: int = 5000, seed: int = 12345) -> float:
    """Monte Carlo ground-truth marginal Psi(a=k) under the DGP with intervention."""
    rng = np.random.default_rng(seed)
    N, T, K = M, cfg.T, cfg.K
    Z = np.zeros((N, T))
    L = np.zeros((N, T))
    Z[:, 0] = rng.normal(0.0, 1.0, N)
    survived = np.ones(N, dtype=bool)
    cum = np.zeros(N)
    a_norm = float(k) / max(K - 1, 1)
    for t in range(T):
        if t > 0:
            Z[:, t] = cfg.psi * Z[:, t - 1] + rng.normal(0.0, cfg.sigma_Z, N)
            L[:, t] = (
                0.5 * Z[:, t] + 0.3 * L[:, t - 1] + rng.normal(0.0, cfg.sigma_L, N)
            )
        else:
            L[:, t] = 0.5 * Z[:, t] + rng.normal(0.0, cfg.sigma_L, N)
        h_t = _expit(
            cfg.hazard_intercept + cfg.beta_A * a_norm
            + cfg.beta_L * L[:, t] + cfg.gamma * Z[:, t]
        )
        cum = cum + survived * h_t
        survived = survived & (rng.uniform(size=N) >= h_t)
    return float(cum.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.0, 0.2, 0.5, 0.8])
    parser.add_argument("--n-reps", type=int, default=500)
    parser.add_argument("--n-per-rep", type=int, default=500,
                        help="Sample size per replicate.")
    parser.add_argument("--T", type=int, default=28)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=1)
    parser.add_argument("--vem-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-bins", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=["standard", "xu", "vem"])
    parser.add_argument("--inner-bootstrap", type=int, default=2,
                        help="Inner refit bootstrap iterations per replicate "
                             "(needed for valid coverage; >=20 recommended).")
    parser.add_argument("--inner-refit", action="store_true",
                        help="Use refit=True for inner bootstrap (slower but "
                             "valid coverage).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_bins = args.target_bins or list(range(args.K))
    print(f"Scenarios: gammas={args.gammas}, n_reps={args.n_reps}, "
          f"N={args.n_per_rep}, T={args.T}, K={args.K}")

    methods_factories = {
        "standard": lambda: StandardGFormula(),
        "xu": lambda: XuGLMM(n_b_draws=50),
        "vem": lambda: VEMSSMBenchmark(VEMConfig(training=TrainingConfig(
            n_epochs=args.vem_epochs, learning_rate=1e-2, n_mc_samples=2,
        ))),
    }

    out: dict[str, np.ndarray] = {}
    md_rows: list[list[str]] = []

    for gamma in args.gammas:
        print(f"\n=== gamma = {gamma} ===")
        base_cfg = DGPConfig(
            N=args.n_per_rep, T=args.T, K=args.K, gamma=gamma, seed=args.seed,
        )
        psi_true = np.array([_true_psi(base_cfg, k) for k in target_bins])
        print(f"  true Psi: {dict(zip(target_bins, psi_true.round(3)))}")

        for m_name in args.methods:
            factory = methods_factories[m_name]
            ests = np.full((args.n_reps, len(target_bins)), np.nan)
            covs = np.zeros((args.n_reps, len(target_bins)), dtype=bool)
            for r in range(args.n_reps):
                cfg = DGPConfig(**{**base_cfg.__dict__, "seed": args.seed + r * 7})
                sample = simulate_tv_latent_confounding(cfg)
                cohort = _sample_to_cohort(sample, K=args.K)
                m = factory()
                m.fit(cohort)
                res = m.dose_response(
                    cohort, target_bins=target_bins,
                    n_bootstrap=max(args.n_bootstrap, 2), seed=args.seed + r,
                    refit=False,
                )
                ests[r] = res.risk_mean
                covs[r] = (res.risk_ci_low <= psi_true) & (psi_true <= res.risk_ci_high)
                if (r + 1) % max(1, args.n_reps // 10) == 0:
                    print(f"    {m_name} rep {r + 1}/{args.n_reps}")

            bias = np.nanmean(ests - psi_true, axis=0)
            ese = np.nanstd(ests, axis=0, ddof=1)
            rmse = np.sqrt(np.nanmean((ests - psi_true) ** 2, axis=0))
            cov_pct = covs.mean(axis=0) * 100.0
            key_prefix = f"gamma={gamma}__{m_name}"
            out[f"{key_prefix}__bias"] = bias
            out[f"{key_prefix}__ese"] = ese
            out[f"{key_prefix}__rmse"] = rmse
            out[f"{key_prefix}__coverage_pct"] = cov_pct
            out[f"{key_prefix}__est_raw"] = ests
            out[f"{key_prefix}__psi_true"] = psi_true
            md_rows.append([
                f"{gamma:.1f}", m_name,
                ", ".join(f"{b:+.3f}" for b in bias),
                ", ".join(f"{s:.3f}" for s in ese),
                ", ".join(f"{r:.3f}" for r in rmse),
                ", ".join(f"{c:.0f}" for c in cov_pct),
            ])

    npz_path = args.out_dir / "table_s1_raw.npz"
    np.savez(npz_path, target_bins=np.array(target_bins, dtype=int), **out)

    md = [
        "# Table S1. Simulation: bias / SE / RMSE / 95% coverage by gamma and method",
        "",
        f"_DGP: per-day hazard with AR(1) latent confounder, T={args.T}, K={args.K}, "
        f"N={args.n_per_rep}/replicate, n_reps={args.n_reps}. "
        f"Bins: {target_bins}. Metrics reported per bin (comma-separated)._",
        "",
        "| gamma | method | bias | ESE | RMSE | coverage % |",
        "|---|---|---|---|---|---|",
    ]
    for row in md_rows:
        md.append("| " + " | ".join(row) + " |")
    md_path = args.out_dir / "table_s1_simulation.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {md_path}\nWrote {npz_path}")


if __name__ == "__main__":
    main()
