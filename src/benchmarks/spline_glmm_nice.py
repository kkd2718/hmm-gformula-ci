"""Spline-RE NICE g-formula benchmark.

Combines:
1. NICE g-formula (Robins 1986) — pooled L equations + forward L simulation
   under intervention, identical to StandardGFormula.
2. Spline-basis time-varying random effect on outcome — fitted via Laplace
   marginal likelihood (Pinheiro-Bates style hierarchical GLMM).

K = 1 reduces to scalar random-intercept NICE g-formula (the existing
"hierarchical g-computation" reference baseline; e.g., McCulloch 2008,
Daniels-Hogan 2008 ch.11). K >= 3 uses natural cubic spline basis for
time-varying RE (proposed extension).

Counterfactual procedure
------------------------
For each MC draw m = 1, ..., M:
  For each subject i:
    Draw  b_i^(m) ~ N(0, Σ̂_b)        [K-dim multivariate normal]
    L_{i,0}^* = L_{i,0}^obs (baseline kept, NICE convention)
    Forward simulate L_{i,t}^* under intervention A^* via pooled L regressions
    For t = 0, ..., T-1:
      logit_t = β̂_0 + β̂_A^T a^*_t + η̂^T L_{i,t}^* + ξ̂^T V_i
                  + b_i^(m) ⋅ B(t) + β̂_time · (t/(T-1))
      cum_i^(m) += survived_i^(m) · sigmoid(logit_t);  survived *= 1 - sigmoid

R̂(a^*) = mean over (i, m) of cum_i^(m)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..data.ards import ARDSCohort
from ..models.spline_glmm import SplineGLMM, SplineGLMMConfig, natural_cubic_basis
from ..training.laplace_em import (
    LaplaceTrainingConfig, train_spline_glmm, laplace_marginal_logll,
)
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from ._resample import cluster_bootstrap_indices, slice_cohort
from .standard_gformula import _fit_linear  # reuse pooled L regressor


# ----------------------------------------------------------------------
# Pooled L equation fitting (numpy; mirrors StandardGFormula)
# ----------------------------------------------------------------------
def _fit_pooled_L(
    L_dyn: np.ndarray, A_bin: np.ndarray, C_static: np.ndarray,
    at_risk: np.ndarray, T: int, l2: float = 1e-4,
) -> tuple[list[np.ndarray], list[float]]:
    """Pooled linear regression L_t | L_{t-1}, A_{t-1}, V, t for each L_j.

    Returns lists (beta_L_j, sd_L_j) per dynamic dim j.
    """
    p_dyn = L_dyn.shape[-1]
    rows, targets, weights = [], [], []
    for t in range(1, T):
        N = L_dyn.shape[0]
        bias = np.ones((N, 1), dtype=np.float64)
        t_col = np.full((N, 1), t / max(T - 1, 1), dtype=np.float64)
        X_t = np.concatenate(
            [bias, L_dyn[:, t - 1, :], A_bin[:, t - 1, :], C_static, t_col],
            axis=1,
        ).astype(np.float64)
        w_t = (at_risk[:, t - 1] * at_risk[:, t]).astype(np.float64)
        rows.append(X_t)
        targets.append(L_dyn[:, t, :])
        weights.append(w_t)
    X = np.vstack(rows)
    Y = np.vstack(targets)
    w = np.concatenate(weights)
    betas, sds = [], []
    for j in range(p_dyn):
        beta_j, sd_j = _fit_linear(X, Y[:, j], w, l2=l2)
        betas.append(beta_j)
        sds.append(max(sd_j, 1e-6))
    return betas, sds


def _build_history_features(
    L_prev: np.ndarray, A_prev: np.ndarray, C: np.ndarray,
    t_idx: int, T: int,
) -> np.ndarray:
    N = L_prev.shape[0]
    bias = np.ones((N, 1), dtype=np.float64)
    t_col = np.full((N, 1), t_idx / max(T - 1, 1), dtype=np.float64)
    return np.concatenate([bias, L_prev, A_prev, C, t_col], axis=1)


# ----------------------------------------------------------------------
# Benchmark wrapper
# ----------------------------------------------------------------------
@dataclass
class SplineGLMMNICEConfig:
    """Hyperparameters for SplineGLMMNICEBenchmark."""
    knots: tuple[float, ...] = (0.0, 3.0, 7.0, 14.0, 21.0)
    n_b_draws: int = 200
    l2_L: float = 1e-4
    training: LaplaceTrainingConfig = field(default_factory=LaplaceTrainingConfig)
    n_mc_subjects: int | None = None
    seed_init: int = 0
    use_gpu: bool = True


class SplineGLMMNICEBenchmark(BenchmarkMethod):
    """Spline-RE NICE g-formula (or scalar-RE if K=1).

    Pipeline:
      fit():
        1. Fit pooled L equations (numpy linear regressions, same as Standard).
        2. Fit GLMM on outcome via Laplace marginal likelihood (PyTorch).
      _counterfactual_risk(k):
        Forward simulate L under A=k, marginalize over b_i ~ N(0, Σ̂_b).
    """

    method_name = "spline_glmm_nice"

    def __init__(self, config: SplineGLMMNICEConfig | None = None) -> None:
        self.config = config or SplineGLMMNICEConfig()
        # State after fit:
        self._beta_L: list[np.ndarray] = []
        self._sd_L: list[float] = []
        self._model: SplineGLMM | None = None
        self._device: torch.device | None = None
        # Cached fixed-effect coefficients in numpy (for fast NICE simulation)
        self._beta_0: float = 0.0
        self._beta_A: np.ndarray | None = None
        self._eta: np.ndarray | None = None
        self._xi: np.ndarray | None = None
        self._beta_time: float = 0.0
        self._B_basis: np.ndarray | None = None        # (T, K_re)
        self._L_chol_np: np.ndarray | None = None      # (K_re, K_re)
        self._n_bins: int = 0
        self._n_dyn: int = 0
        self._n_static: int = 0
        self._t_max: int = 0

    # --------------------------- fit ---------------------------
    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        L = cohort.feature_layout
        K_A, p_dyn, p_stat = L["n_bins"], L["n_dyn"], L["n_static"]
        T = cohort.Y.shape[1]
        self._n_bins = K_A
        self._n_dyn = p_dyn
        self._n_static = p_stat
        self._t_max = T

        # 1) Pooled L equations
        L_dyn = cohort.L_dyn.numpy().astype(np.float64)
        A_bin = cohort.A_bin.numpy().astype(np.float64)
        C_static = cohort.C_static.numpy().astype(np.float64)
        at_risk = cohort.at_risk.numpy().astype(np.float64).squeeze(-1)
        self._beta_L, self._sd_L = _fit_pooled_L(
            L_dyn, A_bin, C_static, at_risk, T, l2=self.config.l2_L,
        )

        # 2) GLMM on outcome via Laplace
        device = (
            torch.device("cuda")
            if (self.config.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )
        self._device = device
        torch.manual_seed(self.config.seed_init)

        ssm_cfg = SplineGLMMConfig(
            n_bins=K_A, n_dyn=p_dyn, n_static=p_stat,
            knots=self.config.knots, t_max=T,
        )
        model = SplineGLMM(ssm_cfg).to(device)

        Y = cohort.Y.to(device)
        A = cohort.A_bin.to(device)
        L_dyn_t = cohort.L_dyn.to(device)
        V = cohort.C_static.to(device)
        at_risk_t = cohort.at_risk.to(device)
        t_norm = cohort.t_norm.to(device)

        train_spline_glmm(
            model, Y, A, L_dyn_t, V, at_risk_t, t_norm,
            config=self.config.training, verbose=False,
        )
        self._model = model

        # Cache numpy versions for fast NICE simulation
        with torch.no_grad():
            self._beta_0 = float(model.beta_0.item())
            self._beta_A = model.beta_A.detach().cpu().numpy().astype(np.float64)
            self._eta = (
                model.eta.detach().cpu().numpy().astype(np.float64)
                if model.eta is not None else None
            )
            self._xi = (
                model.xi.detach().cpu().numpy().astype(np.float64)
                if model.xi is not None else None
            )
            self._beta_time = float(model.beta_time.item())
            self._B_basis = model.B.detach().cpu().numpy().astype(np.float64)
            self._L_chol_np = model.L_chol().detach().cpu().numpy().astype(np.float64)

    # --------------------------- counterfactual ---------------------------
    def _simulate_counterfactual_risk(
        self, cohort: ARDSCohort, intervene_bin: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Per-subject 28-day cumulative incidence under A=intervene_bin."""
        K_A, p_dyn, p_stat, T = (
            self._n_bins, self._n_dyn, self._n_static, self._t_max,
        )
        N_obs = cohort.L_dyn.shape[0]
        M = self.config.n_mc_subjects or N_obs

        L_obs = cohort.L_dyn.numpy().astype(np.float64)
        C_obs = cohort.C_static.numpy().astype(np.float64)

        # Sample baseline (L_0, C) from observed empirical distribution
        idx0 = rng.integers(0, N_obs, size=M)
        L_t = L_obs[idx0, 0, :].copy()
        C_mc = C_obs[idx0]

        A_onehot = np.zeros((M, K_A), dtype=np.float64)
        A_onehot[:, intervene_bin] = 1.0

        # Average over n_b_draws of b_i ~ N(0, Σ̂_b)
        risks_per_draw = []
        for _ in range(self.config.n_b_draws):
            # Sample b_i for each MC subject (shape M, K_re)
            K_re = self._B_basis.shape[1]
            z = rng.standard_normal(size=(M, K_re))
            # Sample x ~ N(0, Σ_b) via x = z @ L_chol.T  where Σ_b = L_chol L_chol^T
            b = z @ self._L_chol_np.T
            # b @ B(t)^T: shape (M, T) — random component per (subject, time)
            random_logit_full = b @ self._B_basis.T   # (M, T)

            # Forward simulate L and accumulate hazard
            L_cur = L_t.copy()
            survived = np.ones(M, dtype=np.float64)
            cum = np.zeros(M, dtype=np.float64)

            for t in range(T):
                if t > 0:
                    X_hist = _build_history_features(
                        L_cur, A_onehot, C_mc, t_idx=t, T=T,
                    )
                    L_new = np.empty_like(L_cur)
                    for j in range(p_dyn):
                        mu_j = X_hist @ self._beta_L[j]
                        L_new[:, j] = mu_j + rng.normal(0.0, self._sd_L[j], size=M)
                    L_cur = L_new
                # Outcome logit
                eta_fix = self._beta_0 + (A_onehot * self._beta_A).sum(axis=-1)
                if self._eta is not None and p_dyn > 0:
                    eta_fix = eta_fix + L_cur @ self._eta
                if self._xi is not None and p_stat > 0:
                    eta_fix = eta_fix + C_mc @ self._xi
                eta_fix = eta_fix + self._beta_time * (t / max(T - 1, 1))
                logit = eta_fix + random_logit_full[:, t]
                p_t = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
                cum = cum + survived * p_t
                survived = survived * (1.0 - p_t)
            risks_per_draw.append(cum)

        return np.mean(np.stack(risks_per_draw, axis=0), axis=0)  # (M,)

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 100, seed: int = 0, refit: bool = True,
    ) -> DoseResponseResult:
        rng = np.random.default_rng(seed)
        if not refit and self._model is None:
            self.fit(cohort)
        K = len(target_bins)
        risk_mat = np.zeros((K, n_bootstrap), dtype=np.float64)
        for b in range(n_bootstrap):
            idx = cluster_bootstrap_indices(cohort.subject_ids, rng)
            boot_cohort = slice_cohort(cohort, idx)
            if refit:
                self.fit(boot_cohort)
            for ki, k in enumerate(target_bins):
                cum = self._simulate_counterfactual_risk(
                    boot_cohort, intervene_bin=k, rng=rng,
                )
                risk_mat[ki, b] = float(cum.mean())
        return DoseResponseResult(
            bins=list(target_bins),
            bin_centers_J_min=bin_centers_J_min(cohort),
            risk_mean=risk_mat.mean(axis=1),
            risk_ci_low=np.quantile(risk_mat, 0.025, axis=1),
            risk_ci_high=np.quantile(risk_mat, 0.975, axis=1),
            risk_raw=risk_mat,
            method_name=self.method_name,
        )
