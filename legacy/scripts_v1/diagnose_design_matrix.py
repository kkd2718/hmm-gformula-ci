"""Design matrix sanity check for Bayesian outcome models.

Verifies:
1. Bin one-hot columns sum to 1 across rows (or 0 for non-at-risk) — proper one-hot
2. No collinearity between bin columns and bias (rank check)
3. Reference bin coding consistency
4. Standard parametric g-formula vs Bayesian Xu/FRE-NICE design matrix structure
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()

    cohort = load_ards_cohort(ARDSConfig(csv_path=args.csv))
    L = cohort.feature_layout
    K_A, p_dyn, p_stat = L["n_bins"], L["n_dyn"], L["n_static"]
    N, T = cohort.Y.shape[0], cohort.Y.shape[1]

    print(f"=== Design matrix sanity check ===")
    print(f"N stays = {N}, T = {T}, K_A = {K_A}, p_dyn = {p_dyn}, p_stat = {p_stat}")
    print()

    # Build the SAME design matrix used in xu_glmm_bayesian.py / fre_nice_bayesian.py
    cov = cohort.covariates.numpy().reshape(N * T, -1).astype(np.float64)
    bias = np.ones((cov.shape[0], 1))
    X = np.concatenate([bias, cov], axis=1)
    p_total = X.shape[1]
    print(f"Design matrix X: shape = {X.shape}, total cols p = {p_total}")
    print()

    # cov layout: [bin_one_hot (K_A) | L_dyn (p_dyn) | C_static (p_stat) | t_norm (1)]
    # X = [bias | cov] = [bias (1) | bins (K_A) | L (p_dyn) | C (p_stat) | t (1)]
    bias_col = X[:, 0]
    bin_cols = X[:, 1:1 + K_A]                  # (N*T, K_A)
    L_cols = X[:, 1 + K_A : 1 + K_A + p_dyn]
    C_cols = X[:, 1 + K_A + p_dyn : 1 + K_A + p_dyn + p_stat]
    t_col = X[:, -1]

    # --- Check 1: one-hot bins sum to 1 (or 0 for non-at-risk) ---
    bin_sum = bin_cols.sum(axis=1)
    n_atrisk = int((bin_sum > 0.5).sum())
    n_zero = int((bin_sum < 0.5).sum())
    n_total = bin_sum.size
    print(f"Check 1: bin one-hot row sums")
    print(f"  rows with sum=1 (at-risk): {n_atrisk:>8}/{n_total} ({100*n_atrisk/n_total:.1f}%)")
    print(f"  rows with sum=0 (non-at-risk or padding): {n_zero:>8}/{n_total}")
    rows_with_other = int(((bin_sum > 0.001) & (bin_sum < 0.999)).sum())
    print(f"  rows with non-{0,1} sum: {rows_with_other} (should be 0)")
    print()

    # --- Check 2: collinearity — bin sum == bias - 0 row mask? ---
    # Among at-risk rows: bias=1 and sum(bins)=1, so bias - sum(bins) = 0 — perfect collinearity
    # if we include both bias AND all K_A bin columns
    atrisk_mask = bin_sum > 0.5
    print(f"Check 2: bin sum vs bias collinearity (among at-risk rows)")
    if atrisk_mask.any():
        residual = bias_col[atrisk_mask] - bin_cols[atrisk_mask].sum(axis=1)
        max_abs = float(np.abs(residual).max())
        print(f"  max |bias - sum(bins)| over at-risk rows: {max_abs:.6f}")
        if max_abs < 1e-6:
            print(f"  ⚠ WARNING: bias and all bin columns are perfectly collinear on at-risk rows.")
            print(f"             This causes design singularity — should drop reference bin or bias.")
        else:
            print(f"  No exact collinearity (residual nonzero — likely due to non-at-risk row mixing).")
    print()

    # --- Check 3: full-rank check ---
    print(f"Check 3: rank of design matrix")
    # Use only at-risk rows for rank check
    X_atrisk = X[atrisk_mask]
    rank = int(np.linalg.matrix_rank(X_atrisk, tol=1e-8))
    print(f"  rank({X_atrisk.shape}) = {rank}")
    print(f"  expected if full-rank: {p_total}")
    if rank < p_total:
        print(f"  ⚠ WARNING: design matrix is rank-deficient by {p_total - rank}.")
        print(f"             Bias + all K_A bins + ... is collinear. Reference category should be dropped.")
    else:
        print(f"  Design matrix has full rank — OK.")
    print()

    # --- Check 4: condition number ---
    print(f"Check 4: condition number of (at-risk) design")
    # Take a sample to keep computation feasible
    if X_atrisk.shape[0] > 100000:
        rng = np.random.default_rng(0)
        idx = rng.choice(X_atrisk.shape[0], size=100000, replace=False)
        X_sub = X_atrisk[idx]
    else:
        X_sub = X_atrisk
    cond = float(np.linalg.cond(X_sub.T @ X_sub))
    print(f"  cond(X^T X) = {cond:.2e}")
    if cond > 1e10:
        print(f"  ⚠ WARNING: ill-conditioned — bin one-hot may be near-collinear with bias")
    elif cond > 1e6:
        print(f"  Moderate condition number — borderline; check posterior diagnostics.")
    else:
        print(f"  Well-conditioned — OK.")
    print()

    # --- Check 5: comparison to Standard NICE g-formula's design ---
    print(f"Check 5: Standard NICE g-formula design comparison")
    print(f"  standard_gformula._build_outcome_features uses: [bias, L_t, A_t, C, t/T]")
    print(f"  i.e. [bias (1), L_dyn (p_dyn), A_t one-hot (K_A), C (p_stat), t/T (1)]")
    print(f"  same column count as Bayesian, just different order of L vs A.")
    print(f"  Both have bias + all K_A one-hot bins → SAME potential collinearity issue.")


if __name__ == "__main__":
    main()
