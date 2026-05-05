"""Diagnostic: is Xu's sigma_b collapsing due to MoM bug or genuine data signal?

Compares three sigma_b update schemes within the same Newton-EM outer loop:
  (1) MoM (current Xu implementation):   sigma2 = var(b_hat)
  (2) EM correct (Pinheiro-Bates 2000):  sigma2 = mean(b_hat^2 + 1/H_ii)
  (3) Profile likelihood (search):       sigma2 = argmax marginal LL

Prints sigma_b trajectory across outer iterations. If (2) or (3) recovers
non-trivial sigma_b while (1) collapses to floor, the current implementation
has a fitting bug, NOT Xu's framework per se.
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


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _build_design(cohort):
    L = cohort.feature_layout
    K = L["n_bins"]
    N, T = cohort.Y.shape[0], cohort.Y.shape[1]
    cov = cohort.covariates.numpy().reshape(N * T, -1)
    y = cohort.Y.numpy().reshape(N * T)
    m = cohort.at_risk.numpy().reshape(N * T)
    bias = np.ones((cov.shape[0], 1), dtype=np.float64)
    X = np.concatenate([bias, cov.astype(np.float64)], axis=1)
    group = np.repeat(cohort.subject_ids, T)
    return X, y.astype(np.float64), m.astype(np.float64), group


def fit_with_sigma_update(X, y, w, group, sigma_method: str,
                          l2: float = 1e-4, max_outer: int = 30,
                          verbose: bool = True):
    """Newton-EM outer loop with configurable sigma^2 update."""
    n, p = X.shape
    groups, inv = np.unique(group, return_inverse=True)
    G = len(groups)
    beta = np.zeros(p, dtype=np.float64)
    b = np.zeros(G, dtype=np.float64)
    sigma2 = 1.0
    sigma_traj = []

    for it in range(max_outer):
        eta = X @ beta + b[inv]
        mu = _expit(eta)
        s = mu * (1.0 - mu) + 1e-8
        W = w * s
        # Newton on beta
        grad_beta = X.T @ (w * (mu - y)) + l2 * beta
        H_beta = (X.T * W) @ X + l2 * np.eye(p)
        beta = beta - np.linalg.solve(H_beta, grad_beta)
        # Newton on b (per-group)
        eta = X @ beta + b[inv]
        mu = _expit(eta)
        s = mu * (1.0 - mu) + 1e-8
        resid = w * (mu - y)
        Wb = w * s
        resid_sum = np.bincount(inv, weights=resid, minlength=G)
        Wb_sum = np.bincount(inv, weights=Wb, minlength=G)
        num = -(resid_sum + b / sigma2)
        den = np.maximum(Wb_sum + 1.0 / sigma2, 1e-8)
        b = b + num / den
        # Per-group Hessian (1/sigma^2 + Wb_sum) — this IS the posterior precision
        H_b = Wb_sum + 1.0 / sigma2
        post_var = 1.0 / np.maximum(H_b, 1e-8)
        # Sigma update — three variants
        if sigma_method == "MoM":
            # Current Xu code: just var(b_hat)
            sigma2_new = max(np.var(b), 1e-4)
        elif sigma_method == "EM":
            # Proper EM: E[b^2 | y] = b_hat^2 + posterior_var
            sigma2_new = max(np.mean(b ** 2 + post_var), 1e-4)
        elif sigma_method == "EM_unbiased":
            # EM divided by (G-1) for unbiased estimate (small G correction)
            sigma2_new = max(np.sum(b ** 2 + post_var) / max(G - 1, 1), 1e-4)
        else:
            raise ValueError(sigma_method)
        sigma2 = sigma2_new
        sigma_traj.append(np.sqrt(sigma2))
        if verbose:
            print(f"  iter {it+1:2d}: sigma_b = {np.sqrt(sigma2):.4f}, "
                  f"|b_hat| max = {np.abs(b).max():.3f}, "
                  f"mean post_var = {post_var.mean():.4f}")

    return beta, b, sigma_traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-outer", type=int, default=30)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_ards_cohort(ARDSConfig(csv_path=args.csv))
    X, y, w, group = _build_design(cohort)
    G = len(np.unique(group))
    print(f"Cohort: N rows = {len(y)}, G subjects = {G}")
    print()

    print("=== (1) MoM (current Xu implementation) ===")
    _, _, traj_mom = fit_with_sigma_update(X, y, w, group, "MoM",
                                            max_outer=args.max_outer)
    print()
    print("=== (2) EM with posterior variance (Pinheiro-Bates) ===")
    _, _, traj_em = fit_with_sigma_update(X, y, w, group, "EM",
                                           max_outer=args.max_outer)
    print()
    print("=== (3) EM unbiased (G-1 denominator) ===")
    _, _, traj_emu = fit_with_sigma_update(X, y, w, group, "EM_unbiased",
                                            max_outer=args.max_outer)

    print("\n=== Comparison summary ===")
    print(f"  Final sigma_b: MoM={traj_mom[-1]:.4f}, "
          f"EM={traj_em[-1]:.4f}, EM_unbiased={traj_emu[-1]:.4f}")

    np.savez(
        args.out_dir / "xu_sigma_diagnostic.npz",
        traj_mom=np.array(traj_mom),
        traj_em=np.array(traj_em),
        traj_em_unbiased=np.array(traj_emu),
    )

    md_lines = [
        "# Xu sigma_b update diagnostic",
        "",
        f"Cohort: G = {G} unique subjects, {args.max_outer} outer iterations.",
        "",
        "| Iter | MoM (current) | EM (proper) | EM unbiased |",
        "|---|---|---|---|",
    ]
    for it in range(len(traj_mom)):
        md_lines.append(
            f"| {it+1} | {traj_mom[it]:.4f} | {traj_em[it]:.4f} | "
            f"{traj_emu[it]:.4f} |"
        )
    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "- If MoM collapses to floor (1e-4 → sigma_b = 0.01) but EM/EM_unbiased "
        "find non-trivial sigma_b → current implementation has fitting bug.",
        "- If all three give similar small values → genuine weak RE signal in data.",
    ])
    (args.out_dir / "xu_sigma_diagnostic.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    print(f"\nWrote {args.out_dir / 'xu_sigma_diagnostic.md'}")


if __name__ == "__main__":
    main()
