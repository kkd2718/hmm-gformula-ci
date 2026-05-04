"""Natural-course validation for Bayesian methods (K=1, K=5 FRE-NICE, Xu).

Forward-simulates each method's generative model under the OBSERVED treatment
trajectory (no intervention), and compares marginal 28-day mortality to the
cohort raw mortality. A well-calibrated method should produce marginal mortality
within bootstrap-of-cohort-rate range of cohort raw (Taubman 2009 IJE).

For Xu Bayesian: uses observed L_t (MSM-style natural course)
For K=1, K=5 FRE-NICE: forward-simulates L_t under observed A_t (NICE-style natural course)

If Standard NICE shows ~25.6% (cohort raw), but K=1 NICE shows e.g. 35%, that
flags a calibration miss in our forward L simulation — could indicate
implementation issue.
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def natural_course_xu_bayesian(cohort, posterior, n_b_draws: int = 50,
                                n_posterior_subset: int = 200,
                                seed: int = 0) -> tuple[float, np.ndarray]:
    """Xu Bayesian natural course: observed L plug-in, observed A, marginalize over b."""
    rng = np.random.default_rng(seed)
    L = cohort.feature_layout
    K_A = L["n_bins"]
    N, T = cohort.Y.shape[0], cohort.Y.shape[1]
    cov = cohort.covariates.numpy().reshape(N * T, -1).astype(np.float64)
    bias = np.ones((cov.shape[0], 1), dtype=np.float64)
    X = np.concatenate([bias, cov], axis=1)               # (N*T, p)

    S_total = posterior["beta"].shape[0]
    S = min(n_posterior_subset, S_total)
    idx = rng.choice(S_total, size=S, replace=False)
    beta_post = posterior["beta"][idx]
    sigma_b_post = posterior["sigma_b"][idx]

    risks_per_post = []
    for s in range(S):
        beta = beta_post[s]
        sigma_b = sigma_b_post[s]
        eta_no_b = (X @ beta).reshape(N, T)
        risks_b = []
        for _ in range(n_b_draws):
            b = rng.normal(0.0, sigma_b, size=N)
            eta = eta_no_b + b[:, None]
            p = _sigmoid(eta)
            survived = np.ones(N)
            cum = np.zeros(N)
            for t in range(T):
                cum = cum + survived * p[:, t]
                survived = survived * (1.0 - p[:, t])
            risks_b.append(cum.mean())
        risks_per_post.append(float(np.mean(risks_b)))
    arr = np.asarray(risks_per_post)
    return float(arr.mean()), arr


def natural_course_fre_nice(cohort, posterior, B_basis: np.ndarray,
                             beta_L: list[np.ndarray], sd_L: list[float],
                             n_posterior_subset: int = 200,
                             n_b_draws_per_post: int = 5,
                             seed: int = 0) -> tuple[float, np.ndarray]:
    """FRE-NICE natural course: forward-simulate L under OBSERVED A, marginalize over b."""
    rng = np.random.default_rng(seed)
    L = cohort.feature_layout
    K_A, p_dyn, p_stat = L["n_bins"], L["n_dyn"], L["n_static"]
    N, T = cohort.Y.shape[0], cohort.Y.shape[1]
    L_obs = cohort.L_dyn.numpy().astype(np.float64)
    A_bin = cohort.A_bin.numpy().astype(np.float64)
    C_static = cohort.C_static.numpy().astype(np.float64)

    S_total = posterior["beta"].shape[0]
    S = min(n_posterior_subset, S_total)
    post_idx = rng.choice(S_total, size=S, replace=False)
    beta_post = posterior["beta"][post_idx]
    L_chol_post = posterior["L_chol"][post_idx]
    K_re = L_chol_post.shape[-1]

    risks_per_post = []
    for s_idx in range(S):
        beta = beta_post[s_idx]
        L_chol = L_chol_post[s_idx]
        risks_b = []
        for _ in range(n_b_draws_per_post):
            z = rng.standard_normal(size=(N, K_re))
            b_subj = z @ L_chol.T
            random_logit_full = b_subj @ B_basis.T            # (N, T)

            L_cur = L_obs[:, 0, :].copy()
            survived = np.ones(N)
            cum = np.zeros(N)
            for t in range(T):
                if t > 0:
                    bias_h = np.ones((N, 1))
                    t_col_h = np.full((N, 1), t / max(T - 1, 1))
                    X_hist = np.concatenate(
                        [bias_h, L_cur, A_bin[:, t - 1, :], C_static, t_col_h],
                        axis=1,
                    )
                    L_new = np.empty_like(L_cur)
                    for j in range(p_dyn):
                        mu_j = X_hist @ beta_L[j]
                        L_new[:, j] = mu_j + rng.normal(0.0, sd_L[j], size=N)
                    L_cur = L_new
                # Outcome logit using observed A_t (natural course)
                bias_o = np.ones((N, 1))
                t_col_o = np.full((N, 1), t / max(T - 1, 1))
                X_t = np.concatenate(
                    [bias_o, A_bin[:, t, :], L_cur, C_static, t_col_o], axis=1,
                )
                eta_fix = X_t @ beta
                logit = eta_fix + random_logit_full[:, t]
                p_t = _sigmoid(logit)
                cum = cum + survived * p_t
                survived = survived * (1.0 - p_t)
            risks_b.append(cum.mean())
        risks_per_post.append(float(np.mean(risks_b)))
    arr = np.asarray(risks_per_post)
    return float(arr.mean()), arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-posterior-subset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["xu", "K1", "K5"])
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cohort = load_ards_cohort(ARDSConfig(csv_path=args.csv))
    Y = cohort.Y.numpy()
    M = cohort.at_risk.numpy()
    death_per_stay = (Y * M).sum(axis=(1, 2))
    cohort_raw = float((death_per_stay > 0).mean())
    print(f"Cohort raw 28-day mortality: {100*cohort_raw:.2f}%")
    print()

    # Re-fit pooled L equations once (frequentist, same as in fre_nice_bayesian.fit)
    from src.benchmarks.standard_gformula import _fit_linear
    L_dyn = cohort.L_dyn.numpy().astype(np.float64)
    A_bin = cohort.A_bin.numpy().astype(np.float64)
    C_static = cohort.C_static.numpy().astype(np.float64)
    at_risk = cohort.at_risk.numpy().astype(np.float64).squeeze(-1)
    L = cohort.feature_layout
    p_dyn = L["n_dyn"]
    T = cohort.Y.shape[1]
    rows, targets, weights = [], [], []
    for t in range(1, T):
        N = L_dyn.shape[0]
        bias = np.ones((N, 1))
        t_col = np.full((N, 1), t / max(T - 1, 1))
        X_t = np.concatenate(
            [bias, L_dyn[:, t - 1, :], A_bin[:, t - 1, :], C_static, t_col], axis=1,
        )
        w_t = (at_risk[:, t - 1] * at_risk[:, t]).astype(np.float64)
        rows.append(X_t); targets.append(L_dyn[:, t, :]); weights.append(w_t)
    X_L = np.vstack(rows); Y_L = np.vstack(targets); w_L = np.concatenate(weights)
    beta_L, sd_L = [], []
    for j in range(p_dyn):
        b_j, s_j = _fit_linear(X_L, Y_L[:, j], w_L, l2=1e-4)
        beta_L.append(b_j); sd_L.append(max(s_j, 1e-6))

    results = {}

    if "xu" in args.methods:
        post_path = args.results_dir / "posterior_xu_bayesian.npz"
        if post_path.exists():
            print("=== Xu Bayesian natural course (observed L plug-in) ===")
            posterior = dict(np.load(post_path))
            mean, arr = natural_course_xu_bayesian(
                cohort, posterior,
                n_posterior_subset=args.n_posterior_subset, seed=args.seed,
            )
            print(f"  posterior-mean NC mortality: {100*mean:.2f}%")
            print(f"  95% CI: ({100*np.quantile(arr, 0.025):.2f}, {100*np.quantile(arr, 0.975):.2f})")
            print(f"  cohort raw: {100*cohort_raw:.2f}%, miss: {100*(mean - cohort_raw):+.2f}%p")
            print()
            results["xu_bayesian"] = (mean, arr)

    from src.models.spline_glmm import natural_cubic_basis

    if "K1" in args.methods:
        post_path = args.results_dir / "posterior_fre_nice_K1.npz"
        if post_path.exists():
            print("=== K=1 FRE-NICE Bayesian natural course (forward L sim, observed A) ===")
            posterior = dict(np.load(post_path))
            B_basis = natural_cubic_basis((14.0,), np.arange(T))
            mean, arr = natural_course_fre_nice(
                cohort, posterior, B_basis, beta_L, sd_L,
                n_posterior_subset=args.n_posterior_subset, seed=args.seed,
            )
            print(f"  posterior-mean NC mortality: {100*mean:.2f}%")
            print(f"  95% CI: ({100*np.quantile(arr, 0.025):.2f}, {100*np.quantile(arr, 0.975):.2f})")
            print(f"  cohort raw: {100*cohort_raw:.2f}%, miss: {100*(mean - cohort_raw):+.2f}%p")
            print()
            results["fre_nice_K1"] = (mean, arr)

    if "K5" in args.methods:
        post_path = args.results_dir / "posterior_fre_nice_K5.npz"
        if post_path.exists():
            print("=== K=5 FRE-NICE Bayesian natural course (forward L sim, observed A) ===")
            posterior = dict(np.load(post_path))
            B_basis = natural_cubic_basis((0.0, 3.0, 7.0, 14.0, 21.0), np.arange(T))
            mean, arr = natural_course_fre_nice(
                cohort, posterior, B_basis, beta_L, sd_L,
                n_posterior_subset=args.n_posterior_subset, seed=args.seed,
            )
            print(f"  posterior-mean NC mortality: {100*mean:.2f}%")
            print(f"  95% CI: ({100*np.quantile(arr, 0.025):.2f}, {100*np.quantile(arr, 0.975):.2f})")
            print(f"  cohort raw: {100*cohort_raw:.2f}%, miss: {100*(mean - cohort_raw):+.2f}%p")
            print()
            results["fre_nice_K5"] = (mean, arr)

    # Markdown report
    md_lines = [
        "# Bayesian Natural Course Validation",
        "",
        f"_Cohort raw 28-day mortality = **{100*cohort_raw:.2f}%**. "
        "Each method forward-simulates under OBSERVED treatment trajectory and "
        "marginalizes over the random effect. A well-calibrated method should "
        "produce mortality close to cohort raw._",
        "",
        "| Method | Mean (%) | 95% CI | Miss (%p) | Calibration |",
        "|---|---|---|---|---|",
    ]
    for name, (mean, arr) in results.items():
        miss = 100 * (mean - cohort_raw)
        ci_lo = 100 * np.quantile(arr, 0.025)
        ci_hi = 100 * np.quantile(arr, 0.975)
        cal = "OK (within ±2%p)" if abs(miss) <= 2.0 else f"miss {abs(miss):.1f}%p"
        md_lines.append(
            f"| {name} | {100*mean:.2f} | ({ci_lo:.2f}, {ci_hi:.2f}) | "
            f"{miss:+.2f} | {cal} |"
        )
    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Xu Bayesian** uses observed L plug-in (no forward L sim), so this NC "
        "is essentially \"factual\" prediction averaged over b ~ N(0, sigma_b^2). "
        "Should be very close to cohort raw if model is well-fit.",
        "- **K=1 / K=5 FRE-NICE** use NICE forward L simulation. Calibration miss "
        "indicates either (a) generative L equations have residual misspecification, "
        "(b) outcome model under model-implied L distribution drifts, or "
        "(c) implementation issue in our forward sim. For Standard parametric "
        "g-formula on this cohort: NC = 24.58% (miss -1%p) → reference baseline.",
        "",
        "If FRE-NICE NC miss is dramatically larger than Standard's miss, the issue "
        "is in our new Bayesian implementation. If similar to Standard, the simulation "
        "is consistent with NICE algorithm.",
    ])
    args.out.write_text("\n".join(md_lines), encoding="utf-8")
    np.savez(
        args.out.with_suffix(".npz"),
        cohort_raw=cohort_raw,
        **{f"{k}_mean": v[0] for k, v in results.items()},
        **{f"{k}_samples": v[1] for k, v in results.items()},
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
