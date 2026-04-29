"""Xu 2024 Bayesian-style g-computation with subject-level random intercept.

Reference: Xu R, Kim S, Hummers L, Shah A, Zeger SL (2024). Causal inference
using multivariate generalized linear mixed-effects models. Biometrics
80(3):ujae100. doi:10.1093/biomtc/ujae100. PMID 39319549.

Identification: any unmeasured confounder is *time-invariant* within
subject and absorbed by the random intercept b_i ~ N(0, sigma_b^2).

Counterfactual marginal risk integrates over the random-effect distribution:
    Psi(a) = E_X [ E_b [ 1 - prod_t (1 - h(X_cf_t, A=a, b)) ] ]
where the inner expectation is taken by Monte Carlo with M draws of
b ~ N(0, sigma_b^2) per subject (per Xu 2024 Bayesian g-computation).
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

from ..data.ards import ARDSCohort
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from ._resample import cluster_bootstrap_indices, slice_cohort


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _laplace_glmm_fit(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, group: np.ndarray,
    l2: float = 1e-4, max_outer: int = 30, tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Logistic GLMM with subject random intercept fit by Laplace approximation.

    Returns (fixed-effect coefficients beta, posterior modes b_i per group,
    sigma_b). Alternates Newton on beta given b, per-group Newton on b_i
    given beta, and method-of-moments update of sigma_b from b_i variance.
    """
    n, p = X.shape
    groups, inv = np.unique(group, return_inverse=True)
    G = len(groups)
    beta = np.zeros(p, dtype=np.float64)
    b = np.zeros(G, dtype=np.float64)
    sigma2 = 1.0
    for _ in range(max_outer):
        eta = X @ beta + b[inv]
        mu = _expit(eta)
        s = mu * (1.0 - mu) + 1e-8
        W = w * s
        grad_beta = X.T @ (w * (mu - y)) + l2 * beta
        H_beta = (X.T * W) @ X + l2 * np.eye(p)
        beta_new = beta - np.linalg.solve(H_beta, grad_beta)
        eta = X @ beta_new + b[inv]
        mu = _expit(eta)
        s = mu * (1.0 - mu) + 1e-8
        resid = w * (mu - y)
        Wb = w * s
        # Vectorized per-group Newton step: O(N) via bincount instead of O(N*G).
        resid_sum = np.bincount(inv, weights=resid, minlength=G)
        Wb_sum = np.bincount(inv, weights=Wb, minlength=G)
        num = -(resid_sum + b / sigma2)
        den = np.maximum(Wb_sum + 1.0 / sigma2, 1e-8)
        b_new = b + num / den
        sigma2_new = max(np.var(b_new), 1e-4)
        if (
            np.max(np.abs(beta_new - beta)) < tol
            and np.max(np.abs(b_new - b)) < tol
        ):
            beta, b, sigma2 = beta_new, b_new, sigma2_new
            break
        beta, b, sigma2 = beta_new, b_new, sigma2_new
    return beta, b, float(np.sqrt(sigma2))


class XuGLMM(BenchmarkMethod):
    """GLMM g-computation with marginal MC integration over b_i (Xu 2024).

    Counterfactual outcome is averaged over n_b_draws random draws from
    the fitted b distribution N(0, sigma_b^2), giving a population-level
    marginal causal estimand consistent with Xu et al. (2024) Biometrics.
    """

    method_name = "xu_glmm"

    def __init__(self, l2: float = 1e-4, n_b_draws: int = 200) -> None:
        self.l2 = l2
        self.n_b_draws = n_b_draws
        self._beta: np.ndarray | None = None
        self._sigma_b: float | None = None

    def _build_design(
        self, cohort: ARDSCohort, override_bin: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        L = cohort.feature_layout
        K = L["n_bins"]
        N, T = cohort.Y.shape[0], cohort.Y.shape[1]
        cov = cohort.covariates.numpy().reshape(N * T, -1)
        y = cohort.Y.numpy().reshape(N * T)
        m = cohort.at_risk.numpy().reshape(N * T)
        if override_bin is not None:
            cov = cov.copy()
            cov[:, :K] = 0.0
            cov[:, override_bin] = 1.0
        bias = np.ones((cov.shape[0], 1), dtype=np.float64)
        X = np.concatenate([bias, cov.astype(np.float64)], axis=1)
        group = np.repeat(cohort.subject_ids, T)
        return X, y.astype(np.float64), m.astype(np.float64), group

    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        X, y, m, group = self._build_design(cohort)
        beta, _, sigma_b = _laplace_glmm_fit(X, y, m, group, l2=self.l2)
        self._beta = beta
        self._sigma_b = sigma_b

    def _counterfactual_risk(
        self, cohort: ARDSCohort, k: int, idx: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        X, _, _, _ = self._build_design(cohort, override_bin=k)
        N, T = cohort.Y.shape[0], cohort.Y.shape[1]
        eta_no_b = (X @ self._beta).reshape(N, T)
        risks_per_draw = []
        for _ in range(self.n_b_draws):
            b_per_subject = rng.normal(0.0, self._sigma_b, size=N)
            eta = eta_no_b + b_per_subject[:, None]
            p = _expit(eta)
            survived = np.ones(N, dtype=np.float64)
            cum = np.zeros(N, dtype=np.float64)
            for t in range(T):
                cum = cum + survived * p[:, t]
                survived = survived * (1.0 - p[:, t])
            risks_per_draw.append(float(cum[idx].mean()))
        return float(np.mean(risks_per_draw))

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 100, seed: int = 0, refit: bool = True,
    ) -> DoseResponseResult:
        """Bootstrap dose-response. refit=True (default) refits beta and
        sigma_b on each resampled cohort for valid uncertainty CIs."""
        rng = np.random.default_rng(seed)
        if not refit and self._beta is None:
            self.fit(cohort)
        risks_per_bin: list[list[float]] = []
        for k in target_bins:
            boot = []
            for _ in range(n_bootstrap):
                idx = cluster_bootstrap_indices(cohort.subject_ids, rng)
                boot_cohort = slice_cohort(cohort, idx)
                if refit:
                    self.fit(boot_cohort)
                idx_local = np.arange(boot_cohort.Y.shape[0])
                boot.append(
                    self._counterfactual_risk(boot_cohort, k, idx_local, rng)
                )
            risks_per_bin.append(boot)
        risk_mat = np.asarray(risks_per_bin)
        return DoseResponseResult(
            bins=list(target_bins),
            bin_centers_J_min=bin_centers_J_min(cohort),
            risk_mean=risk_mat.mean(axis=1),
            risk_ci_low=np.quantile(risk_mat, 0.025, axis=1),
            risk_ci_high=np.quantile(risk_mat, 0.975, axis=1),
            risk_raw=risk_mat,
            method_name=self.method_name,
        )
