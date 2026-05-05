"""VEM-SSM training diagnostics: ELBO trajectory + factual prediction calibration.

Pre-specified hyperparameter selection criteria (must be set BEFORE inspecting
counterfactual outputs to avoid cherry-picking):

    (C1) ELBO plateau:  |ELBO(t) - ELBO(t-50)| / |ELBO(t)| < 0.001 for some t.
    (C2) Factual calibration: held-out 20% test Brier <= reference pooled-logistic
         baseline (or within 5% relative).
    (C3) No NaN / divergence.

A given (init_beta_0, n_epochs, lr) configuration passes if all three hold.
The first passing configuration on a coarse grid is selected.

Outputs:
    <out_dir>/vem_diagnostics.npz
    <out_dir>/vem_elbo_trajectory.png
    <out_dir>/vem_diagnostics.md
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.models.linear_gaussian_ssm import LinearGaussianSSM, SSMConfig
from src.models.variational_posterior import (
    StructuredGaussianMarkovPosterior, QConfig
)
from src.training.variational_em import train_vem, TrainingConfig
from src.benchmarks._resample import slice_cohort


def _train_test_indices(subject_ids: np.ndarray, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    unique_pat = np.unique(subject_ids)
    rng.shuffle(unique_pat)
    n_test_pat = int(round(len(unique_pat) * test_frac))
    test_pat = set(unique_pat[:n_test_pat].tolist())
    test_mask = np.array([s in test_pat for s in subject_ids])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


@torch.no_grad()
def _factual_brier(model, posterior, cohort) -> float:
    """Average Brier score across (stay, t) where at_risk=1, using posterior mean Z."""
    device = next(model.parameters()).device
    A = cohort.A_bin.to(device)
    L = cohort.L_dyn.to(device)
    V = cohort.C_static.to(device)
    Y_dev = cohort.Y.to(device)
    t_norm = cohort.t_norm.to(device)
    Z, _, _ = posterior.sample_trajectory(A=A, L=L, V=V, Y=Y_dev, n_samples=4)
    Z_mean = Z.mean(dim=0)
    N, T, _ = Z_mean.shape
    Z_flat = Z_mean.reshape(N * T, 1)
    A_flat = A.reshape(N * T, -1)
    L_flat = L.reshape(N * T, -1)
    V_rep = V.unsqueeze(1).expand(N, T, V.shape[-1]).reshape(N * T, -1)
    t_flat = t_norm.reshape(N * T, 1)
    p = torch.sigmoid(model.outcome_logit(Z_flat, A_flat, L_flat, V_rep, t_flat).reshape(N, T))
    Y = Y_dev.squeeze(-1)
    M = cohort.at_risk.to(device).squeeze(-1)
    se = ((p - Y) ** 2) * M
    return float(se.sum() / max(M.sum().item(), 1.0))


def _baseline_pooled_logistic_brier(cohort_train, cohort_test) -> float:
    """Reference: pooled logistic with no latent state on training cohort."""
    from src.benchmarks.standard_gformula import _fit_logistic, _expit
    L = cohort_train.feature_layout
    K = L["n_bins"]
    Ntr, T = cohort_train.Y.shape[0], cohort_train.Y.shape[1]
    cov = cohort_train.covariates.numpy().reshape(Ntr * T, -1).astype(np.float64)
    bias = np.ones((cov.shape[0], 1))
    Xtr = np.concatenate([bias, cov], axis=1)
    ytr = cohort_train.Y.numpy().reshape(Ntr * T)
    wtr = cohort_train.at_risk.numpy().reshape(Ntr * T)
    beta = _fit_logistic(Xtr, ytr, wtr, l2=1e-4)
    Nte, _ = cohort_test.Y.shape[0], cohort_test.Y.shape[1]
    cov_te = cohort_test.covariates.numpy().reshape(Nte * T, -1).astype(np.float64)
    Xte = np.concatenate([np.ones((cov_te.shape[0], 1)), cov_te], axis=1)
    pte = _expit(Xte @ beta)
    yte = cohort_test.Y.numpy().reshape(Nte * T)
    mte = cohort_test.at_risk.numpy().reshape(Nte * T)
    se = ((pte - yte) ** 2) * mte
    return float(se.sum() / max(mte.sum(), 1.0))


def _check_plateau(elbo: list[float], window: int, rel_tol: float) -> int | None:
    """Return earliest epoch where |ELBO(t) - ELBO(t - window)| / |ELBO(t)| < rel_tol."""
    for t in range(window, len(elbo)):
        denom = max(abs(elbo[t]), 1e-6)
        if abs(elbo[t] - elbo[t - window]) / denom < rel_tol:
            return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument(
        "--init-beta-0-grid", type=float, nargs="+",
        default=[-3.0, -4.5, -5.5],
    )
    parser.add_argument("--n-epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-mc", type=int, default=4)
    parser.add_argument("--smoothness", type=float, default=0.02)
    parser.add_argument("--plateau-window", type=int, default=50)
    parser.add_argument("--plateau-tol", type=float, default=1e-3)
    parser.add_argument("--brier-tolerance", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ARDSConfig(csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t)
    cohort = load_ards_cohort(cfg)
    train_idx, test_idx = _train_test_indices(
        cohort.subject_ids, args.test_frac, args.seed,
    )
    train_cohort = slice_cohort(cohort, train_idx)
    test_cohort = slice_cohort(cohort, test_idx)
    print(f"[split] train n={len(train_idx)}, test n={len(test_idx)}")

    baseline_brier = _baseline_pooled_logistic_brier(train_cohort, test_cohort)
    print(f"[baseline] pooled-logistic test Brier = {baseline_brier:.5f}")

    layout = train_cohort.feature_layout
    K = layout["n_bins"]

    histories: dict[float, dict] = {}
    summary_rows: list[list[str]] = []
    for init_b0 in args.init_beta_0_grid:
        ssm_cfg = SSMConfig(
            n_bins=K, n_dyn_covariates=layout["n_dyn"],
            n_static_covariates=layout["n_static"], fit_time_effect=True,
            init_beta_0=init_b0,
        )
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LinearGaussianSSM(ssm_cfg).to(device)
        posterior = StructuredGaussianMarkovPosterior(QConfig(
            n_bins=K, n_dyn_covariates=layout["n_dyn"],
            n_static_covariates=layout["n_static"],
        )).to(device)
        tcfg = TrainingConfig(
            n_epochs=args.n_epochs, learning_rate=args.lr,
            n_mc_samples=args.n_mc, smoothness_lambda=args.smoothness,
            grad_clip=5.0, print_every=50,
            early_stop_patience=10**6, early_stop_tol=0.0,
        )
        print(f"\n=== init_beta_0 = {init_b0:+.1f} ===")
        history = train_vem(
            model, posterior,
            Y=train_cohort.Y, A=train_cohort.A_bin, L=train_cohort.L_dyn,
            V=train_cohort.C_static, at_risk=train_cohort.at_risk,
            t_norm=train_cohort.t_norm, config=tcfg, verbose=True,
        )
        plateau_t = _check_plateau(history.elbo, args.plateau_window, args.plateau_tol)
        try:
            test_brier = _factual_brier(model, posterior, test_cohort)
        except Exception:
            test_brier = float("nan")
        passes_C1 = plateau_t is not None
        passes_C2 = (
            np.isfinite(test_brier)
            and test_brier <= baseline_brier * (1.0 + args.brier_tolerance)
        )
        passes_C3 = all(np.isfinite(history.elbo))
        histories[init_b0] = {
            "elbo": np.asarray(history.elbo),
            "ll_y": np.asarray(history.expected_log_lik_y),
            "ll_l": np.asarray(history.expected_log_lik_l),
            "kl": np.asarray(history.kl),
            "test_brier": test_brier,
            "plateau_epoch": plateau_t if plateau_t is not None else -1,
        }
        summary_rows.append([
            f"{init_b0:+.1f}", str(args.n_epochs), f"{args.lr:.0e}",
            f"{history.elbo[-1]:.1f}",
            str(plateau_t) if plateau_t is not None else "—",
            f"{test_brier:.5f}", f"{baseline_brier:.5f}",
            "✓" if (passes_C1 and passes_C2 and passes_C3) else "✗",
        ])

    npz = {f"init={k}__{ki}": v for k, dct in histories.items() for ki, v in dct.items()}
    npz["baseline_brier"] = np.array([baseline_brier])
    np.savez(args.out_dir / "vem_diagnostics.npz", **npz)

    # ELBO trajectory plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    for init_b0, dct in histories.items():
        axes[0].plot(dct["elbo"], label=f"init β₀ = {init_b0:+.1f}")
        axes[1].plot(dct["kl"], label=f"init β₀ = {init_b0:+.1f}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("ELBO")
    axes[0].set_title("ELBO trajectory"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("KL(q||p)")
    axes[1].set_title("KL chain"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "vem_elbo_trajectory.png", dpi=200, bbox_inches="tight")

    md = [
        "# VEM-SSM training diagnostics",
        "",
        f"_Train n = {len(train_idx)}, test n = {len(test_idx)}, "
        f"reference pooled-logistic test Brier = {baseline_brier:.5f}._",
        "",
        f"_Selection criteria_:",
        f"- (C1) ELBO plateau: relative change over {args.plateau_window} "
        f"epochs < {args.plateau_tol}",
        f"- (C2) Held-out test Brier ≤ baseline × (1 + {args.brier_tolerance})",
        f"- (C3) No NaN / divergence",
        "",
        "| init β₀ | n_epochs | lr | final ELBO | plateau epoch | test Brier | baseline Brier | passes all 3? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        md.append("| " + " | ".join(r) + " |")
    (args.out_dir / "vem_diagnostics.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out_dir / 'vem_diagnostics.md'}")


if __name__ == "__main__":
    main()
