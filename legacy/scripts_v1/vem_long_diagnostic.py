"""Long-run VEM diagnostic: track ELBO, KL, and natural-course bias every 25 epochs.

Tests two hypotheses:
(H1) ELBO and NC bias trade off — as ELBO maximizes, NC drifts from cohort raw
(H2) lr=1e-2 is too large; lr=3e-3 with longer training reaches better optimum

For each (cohort, lr) cell:
- Train 500 epochs in 25-epoch chunks
- After each chunk: measure ELBO, KL, factual Brier, natural-course mortality
- Save trajectory + final loss curve
"""
from __future__ import annotations
import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks._resample import cluster_bootstrap_indices, slice_cohort
from src.models.linear_gaussian_ssm import LinearGaussianSSM, SSMConfig
from src.models.variational_posterior import (
    StructuredGaussianMarkovPosterior, QConfig
)
from src.training.variational_em import train_vem, TrainingConfig
from scripts.sanity_check import natural_course_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--total-epochs", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--lr-grid", type=float, nargs="+", default=[3e-3, 1e-2])
    parser.add_argument("--smooth", type=float, default=0.02)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-full-cohort", action="store_true",
                        help="Use full cohort instead of one bootstrap (more comparable to main run).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t,
    ))

    if args.use_full_cohort:
        cohort = full
        print(f"[cohort] FULL cohort, N = {full.Y.shape[0]}")
    else:
        rng = np.random.default_rng(42)
        idx = cluster_bootstrap_indices(full.subject_ids, rng)
        cohort = slice_cohort(full, idx)
        print(f"[cohort] One bootstrap replicate, N = {cohort.Y.shape[0]}")

    # Cohort raw mortality (target for natural-course)
    Y = cohort.Y.numpy()
    M = cohort.at_risk.numpy()
    death_per_stay = (Y * M).sum(axis=(1, 2))
    cohort_raw = float((death_per_stay > 0).mean())
    print(f"[ref] cohort raw 28-day mortality = {cohort_raw * 100:.2f}%")

    n_chunks = args.total_epochs // args.chunk_size
    layout = cohort.feature_layout

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)

    all_records = {}
    for lr in args.lr_grid:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        ssm_cfg = SSMConfig(
            n_bins=layout["n_bins"], n_dyn_covariates=layout["n_dyn"],
            n_static_covariates=layout["n_static"],
            z_depends_on_treatment_lag=True, fit_time_effect=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LinearGaussianSSM(ssm_cfg).to(device)
        posterior = StructuredGaussianMarkovPosterior(QConfig(
            n_bins=layout["n_bins"], n_dyn_covariates=layout["n_dyn"],
            n_static_covariates=layout["n_static"],
        )).to(device)

        records = {
            "epoch": [], "elbo": [], "ll_y": [], "ll_l": [], "kl": [], "nc_pct": [],
        }

        print(f"\n=== lr = {lr:.0e}, total epochs = {args.total_epochs} ===")
        cumulative_epochs = 0
        for c in range(n_chunks):
            tcfg = TrainingConfig(
                n_epochs=args.chunk_size, learning_rate=lr,
                smoothness_lambda=args.smooth, n_mc_samples=4,
                grad_clip=5.0, print_every=10**6,
                early_stop_patience=10**6, early_stop_tol=0.0,
            )
            history = train_vem(
                model, posterior,
                Y=cohort.Y, A=cohort.A_bin, L=cohort.L_dyn, V=cohort.C_static,
                at_risk=cohort.at_risk, t_norm=cohort.t_norm,
                config=tcfg, verbose=False,
            )
            cumulative_epochs += args.chunk_size
            nc = natural_course_simulation(
                model, cohort, n_mc_subjects=cohort.Y.shape[0], seed=args.seed,
            )
            records["epoch"].append(cumulative_epochs)
            records["elbo"].append(history.elbo[-1])
            records["ll_y"].append(history.expected_log_lik_y[-1])
            records["ll_l"].append(history.expected_log_lik_l[-1])
            records["kl"].append(history.kl[-1])
            records["nc_pct"].append(nc * 100)
            print(f"  epoch {cumulative_epochs:4d}: "
                  f"ELBO={history.elbo[-1]:.0f}  "
                  f"ll_Y={history.expected_log_lik_y[-1]:.0f}  "
                  f"ll_L={history.expected_log_lik_l[-1]:.0f}  "
                  f"KL={history.kl[-1]:.0f}  "
                  f"NC={nc * 100:.2f}%")

        all_records[lr] = records
        np.savez(
            args.out_dir / f"trajectory_lr{lr}.npz",
            **{k: np.array(v) for k, v in records.items()},
            cohort_raw=cohort_raw,
        )

    # Plotting
    for lr, rec in all_records.items():
        ep = rec["epoch"]
        axes[0, 0].plot(ep, rec["elbo"], "o-", label=f"lr={lr:.0e}")
        axes[0, 1].plot(ep, rec["nc_pct"], "o-", label=f"lr={lr:.0e}")
        axes[1, 0].plot(ep, rec["ll_y"], "o-", label=f"ll_Y, lr={lr:.0e}")
        axes[1, 0].plot(ep, rec["ll_l"], "s--", label=f"ll_L, lr={lr:.0e}")
        axes[1, 1].plot(ep, rec["kl"], "o-", label=f"lr={lr:.0e}")

    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("ELBO")
    axes[0, 0].set_title("ELBO trajectory"); axes[0, 0].grid(alpha=0.3); axes[0, 0].legend()
    axes[0, 1].axhline(cohort_raw * 100, color="k", ls="--", lw=0.8, label="cohort raw")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("Natural-course mortality (%)")
    axes[0, 1].set_title(f"Natural-course vs cohort raw {cohort_raw * 100:.1f}%")
    axes[0, 1].grid(alpha=0.3); axes[0, 1].legend()
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("log-lik")
    axes[1, 0].set_title("Y vs L emission likelihoods"); axes[1, 0].grid(alpha=0.3); axes[1, 0].legend()
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("KL(q || p)")
    axes[1, 1].set_title("KL chain"); axes[1, 1].grid(alpha=0.3); axes[1, 1].legend()
    fig.suptitle(
        f"VEM long-run diagnostic — total {args.total_epochs} epochs, smooth={args.smooth}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "long_diagnostic.png", dpi=150, bbox_inches="tight")
    print(f"\nWrote {args.out_dir / 'long_diagnostic.png'}")


if __name__ == "__main__":
    main()
