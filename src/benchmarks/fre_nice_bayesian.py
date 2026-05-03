"""Functional Random-Effect NICE g-formula — Bayesian implementation.

Bayesian extension of Robins (1986) parametric g-formula with subject-specific
time-varying random effect on the outcome hazard, parameterized via fixed
natural-cubic-spline basis. Generalizes Xu 2024's scalar random intercept
to a finite-rank Gaussian process random effect on day t.

Generative model
----------------
Outcome:
    beta, eta, xi   ~  N(0, 5^2)
    L_corr          ~  LKJCholesky(K_re, eta = 2)
    tau_k           ~  HalfCauchy(0, 2.5),  k = 1, ..., K_re
    L_chol          =  diag(tau) @ L_corr               (Cholesky of Sigma_b)
    b_i | L_chol    ~  N(0, L_chol L_chol^T)            via non-centered z_i
    logit P(Y_it=1) =  beta_0 + beta_A^T A_it + eta^T L_it + xi^T V_i
                       + b_i^T B(t)

L equations (NICE forward simulation):
    Pooled linear regression (frequentist; same as Standard parametric g-formula)
    fit on observed (L_{t-1}, A_{t-1}, V) -> L_t;  no random effect on L.

Counterfactual under intervention a*
------------------------------------
For each posterior draw s:
    Use (beta^s, eta^s, xi^s, L_chol^s) and per-subject b_i^s
    For each subject i (or sampled MC subjects):
        Sample baseline (L_0, V) from observed empirical
        For t = 1, ..., T-1:
            Forward-sim L_t from pooled regression at A=a*
        For t = 0, ..., T-1:
            logit_t = ... + b_i^s · B(t)  (functional RE contribution)
        Accumulate cumulative hazard.
Posterior mean + 95% credible interval over draws.

Why Bayesian
------------
Consistency with Xu 2024 inference framework (also Bayesian MCMC). LKJ +
half-Cauchy priors prevent the sigma_b collapse seen in frequentist Laplace
+ MoM. Replaces frequentist Laplace ML in spline_glmm_nice.py for primary
analysis (Laplace retained for LOCO sensitivity speed).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from ..data.ards import ARDSCohort
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from .standard_gformula import _fit_linear
from ..models.spline_glmm import natural_cubic_basis


def _fre_nice_model(
    X_outcome: jnp.ndarray, b_basis_per_obs: jnp.ndarray,
    y: jnp.ndarray, mask: jnp.ndarray,
    group_idx: jnp.ndarray, n_groups: int, K_re: int,
) -> None:
    """numpyro model for FRE-NICE outcome layer.

    X_outcome      : (N_obs, p)   design (intercept + A + L + V, no t-effect)
    b_basis_per_obs: (N_obs, K_re) spline basis evaluated at each row's t
    """
    p = X_outcome.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 5.0))
    # Sigma_b parameterized via LKJ correlation + Half-Cauchy scales.
    # K_re == 1 special-case: LKJCholesky requires dim >= 2, fall back to
    # plain HalfCauchy on a scalar (recovering Xu 2024's scalar RE prior).
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.full((K_re,), 2.5)))
    if K_re == 1:
        L_chol = numpyro.deterministic("L_chol", tau.reshape(1, 1))
    else:
        L_corr = numpyro.sample(
            "L_corr", dist.LKJCholesky(K_re, concentration=2.0),
        )
        L_chol = numpyro.deterministic(
            "L_chol", jnp.expand_dims(tau, -1) * L_corr,
        )
    # Non-centered hierarchical: b_i = L_chol @ z_i
    z = numpyro.sample(
        "z_b", dist.Normal(jnp.zeros((n_groups, K_re)), 1.0),
    )
    b = numpyro.deterministic("b", z @ L_chol.T)              # (n_groups, K_re)
    # Per-observation random contribution: b_{group_idx} · B(t_obs)
    re_contrib = jnp.sum(b[group_idx] * b_basis_per_obs, axis=-1)
    logit = X_outcome @ beta + re_contrib
    log_p = mask * dist.Bernoulli(logits=logit).log_prob(y)
    numpyro.factor("loglik", log_p.sum())


@dataclass
class FRENICEBayesianConfig:
    """Hyperparameters for FRE-NICE Bayesian benchmark."""
    knots: tuple[float, ...] = (0.0, 3.0, 7.0, 14.0, 21.0)
    n_warmup: int = 1000
    n_samples: int = 1000
    n_chains: int = 4
    chain_method: str = "parallel"
    target_accept: float = 0.9
    n_posterior_subset: int = 200
    n_b_draws_per_post: int = 5
    n_mc_subjects: int | None = None
    l2_L: float = 1e-4
    seed: int = 0


class FRENICEBayesianBenchmark(BenchmarkMethod):
    """Bayesian FRE-NICE g-formula via numpyro NUTS."""

    method_name = "fre_nice_bayesian"

    def __init__(self, config: FRENICEBayesianConfig | None = None) -> None:
        self.config = config or FRENICEBayesianConfig()
        # State after fit
        self._posterior: dict | None = None
        self._B_basis: np.ndarray | None = None        # (T, K_re)
        self._beta_L: list[np.ndarray] = []
        self._sd_L: list[float] = []
        self._n_groups: int = 0
        self._n_bins: int = 0
        self._n_dyn: int = 0
        self._n_static: int = 0
        self._t_max: int = 0

    # ----- design assembly -----
    def _build_outcome_design(
        self, cohort: ARDSCohort, override_bin: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build outcome-stage features. Returns
        (X_outcome, B_per_obs, y, mask, group_idx)."""
        L = cohort.feature_layout
        K_A = L["n_bins"]
        N, T = cohort.Y.shape[0], cohort.Y.shape[1]
        cov = cohort.covariates.numpy().reshape(N * T, -1)
        y = cohort.Y.numpy().reshape(N * T)
        m = cohort.at_risk.numpy().reshape(N * T)
        if override_bin is not None:
            cov = cov.copy()
            cov[:, :K_A] = 0.0
            cov[:, override_bin] = 1.0
        bias = np.ones((cov.shape[0], 1), dtype=np.float64)
        X_outcome = np.concatenate([bias, cov.astype(np.float64)], axis=1)
        # B basis tiled to each observation's t
        B = self._B_basis                                          # (T, K_re)
        B_per_obs = np.tile(B, (N, 1))                             # (N*T, K_re)
        _, inv = np.unique(cohort.subject_ids, return_inverse=True)
        group_idx = np.repeat(inv, T)
        return (
            X_outcome.astype(np.float64), B_per_obs.astype(np.float64),
            y.astype(np.float64), m.astype(np.float64), group_idx,
        )

    def _build_history_features(
        self, L_prev: np.ndarray, A_prev: np.ndarray, C: np.ndarray,
        t_idx: int, T: int,
    ) -> np.ndarray:
        N = L_prev.shape[0]
        bias = np.ones((N, 1), dtype=np.float64)
        t_col = np.full((N, 1), t_idx / max(T - 1, 1), dtype=np.float64)
        return np.concatenate([bias, L_prev, A_prev, C, t_col], axis=1)

    # ----- fit -----
    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        L = cohort.feature_layout
        K_A, p_dyn, p_stat = L["n_bins"], L["n_dyn"], L["n_static"]
        T = cohort.Y.shape[1]
        self._n_bins, self._n_dyn, self._n_static, self._t_max = (
            K_A, p_dyn, p_stat, T,
        )

        # 1) Spline basis on integer day grid (QR-orthonormalized inside)
        self._B_basis = natural_cubic_basis(
            self.config.knots, np.arange(T),
        )
        K_re = self._B_basis.shape[1]

        # 2) Pooled L equations (frequentist, same as Standard NICE)
        L_dyn = cohort.L_dyn.numpy().astype(np.float64)
        A_bin = cohort.A_bin.numpy().astype(np.float64)
        C_static = cohort.C_static.numpy().astype(np.float64)
        at_risk = cohort.at_risk.numpy().astype(np.float64).squeeze(-1)
        rows, targets, weights = [], [], []
        for t in range(1, T):
            X_t = self._build_history_features(
                L_dyn[:, t - 1, :], A_bin[:, t - 1, :], C_static, t_idx=t, T=T,
            )
            w_t = (at_risk[:, t - 1] * at_risk[:, t]).astype(np.float64)
            rows.append(X_t); targets.append(L_dyn[:, t, :]); weights.append(w_t)
        X_L = np.vstack(rows); Y_L = np.vstack(targets); w_L = np.concatenate(weights)
        self._beta_L, self._sd_L = [], []
        for j in range(p_dyn):
            beta_j, sd_j = _fit_linear(X_L, Y_L[:, j], w_L, l2=self.config.l2_L)
            self._beta_L.append(beta_j)
            self._sd_L.append(max(sd_j, 1e-6))

        # 3) Bayesian outcome fit via NUTS
        X_out, B_per_obs, y, mask, group_idx = self._build_outcome_design(cohort)
        n_groups = int(group_idx.max() + 1)
        self._n_groups = n_groups

        kernel = NUTS(_fre_nice_model, target_accept_prob=self.config.target_accept)
        mcmc = MCMC(
            kernel,
            num_warmup=self.config.n_warmup,
            num_samples=self.config.n_samples,
            num_chains=self.config.n_chains,
            chain_method=self.config.chain_method,
            progress_bar=False,
        )
        rng_key = jr.PRNGKey(self.config.seed)
        mcmc.run(
            rng_key,
            X_outcome=jnp.asarray(X_out),
            b_basis_per_obs=jnp.asarray(B_per_obs),
            y=jnp.asarray(y), mask=jnp.asarray(mask),
            group_idx=jnp.asarray(group_idx),
            n_groups=n_groups, K_re=K_re,
        )
        samples_flat = mcmc.get_samples()
        self._posterior = {
            "beta": np.asarray(samples_flat["beta"]),         # (S, p)
            "L_chol": np.asarray(samples_flat["L_chol"]),     # (S, K_re, K_re)
            "tau": np.asarray(samples_flat["tau"]),           # (S, K_re)
        }
        from numpyro.diagnostics import summary
        try:
            samples_chains = mcmc.get_samples(group_by_chain=True)
            diag = summary(samples_chains, prob=0.95)
            tau_diag = diag.get("tau", {})
            rhat_arr = np.asarray(tau_diag.get("r_hat", np.array([np.nan])))
            r_hat_max = float(rhat_arr.max())
        except Exception:
            r_hat_max = float("nan")
        print(
            f"  [FRE-NICE Bayesian fit] tau (per-basis-dim scale): "
            f"mean = {self._posterior['tau'].mean(axis=0)}, "
            f"R-hat max = {r_hat_max:.3f}"
        )

    # ----- counterfactual forward sim -----
    def _counterfactual_per_bin(
        self, cohort: ARDSCohort, k: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-distributed risk under A=k, returns (S_subset,)."""
        K_A, p_dyn, p_stat, T = (
            self._n_bins, self._n_dyn, self._n_static, self._t_max,
        )
        N_obs = cohort.L_dyn.shape[0]
        M = self.config.n_mc_subjects or N_obs
        L_obs = cohort.L_dyn.numpy().astype(np.float64)
        C_obs = cohort.C_static.numpy().astype(np.float64)

        S_total = self._posterior["beta"].shape[0]
        S = min(self.config.n_posterior_subset, S_total)
        post_idx = rng.choice(S_total, size=S, replace=False)
        beta_post = self._posterior["beta"][post_idx]            # (S, p)
        L_chol_post = self._posterior["L_chol"][post_idx]        # (S, K_re, K_re)
        K_re = L_chol_post.shape[-1]

        # Pre-sample baseline subjects ONCE per dose_response call
        idx0 = rng.integers(0, N_obs, size=M)
        L_t0 = L_obs[idx0, 0, :].copy()
        C_mc = C_obs[idx0]
        A_onehot = np.zeros((M, K_A), dtype=np.float64)
        A_onehot[:, k] = 1.0

        # NICE forward L simulation: same trajectory across posterior draws
        # for paired RD across bins (RNG shared via L_seeds outside this fn).
        # Inner loop: for each posterior draw, draw b ~ N(0, Σ_b^s), accumulate.
        risk_per_post = np.zeros(S, dtype=np.float64)
        for s_idx in range(S):
            beta = beta_post[s_idx]                              # (p,)
            L_chol = L_chol_post[s_idx]                          # (K_re, K_re)
            risks_b = []
            for _ in range(self.config.n_b_draws_per_post):
                z = rng.standard_normal(size=(M, K_re))
                b_subj = z @ L_chol.T                            # (M, K_re)
                random_logit_full = b_subj @ self._B_basis.T     # (M, T)

                # Forward simulate L
                L_cur = L_t0.copy()
                survived = np.ones(M, dtype=np.float64)
                cum = np.zeros(M, dtype=np.float64)
                for t in range(T):
                    if t > 0:
                        X_hist = self._build_history_features(
                            L_cur, A_onehot, C_mc, t_idx=t, T=T,
                        )
                        L_new = np.empty_like(L_cur)
                        for j in range(p_dyn):
                            mu_j = X_hist @ self._beta_L[j]
                            L_new[:, j] = mu_j + rng.normal(
                                0.0, self._sd_L[j], size=M,
                            )
                        L_cur = L_new
                    # Outcome logit (use posterior beta)
                    bias_col = np.ones((M, 1), dtype=np.float64)
                    t_col = np.full((M, 1), t / max(T - 1, 1))
                    X_t = np.concatenate(
                        [bias_col, A_onehot, L_cur, C_mc, t_col], axis=1,
                    )
                    eta_fix = X_t @ beta
                    logit = eta_fix + random_logit_full[:, t]
                    p_t = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
                    cum = cum + survived * p_t
                    survived = survived * (1.0 - p_t)
                risks_b.append(cum.mean())
            risk_per_post[s_idx] = float(np.mean(risks_b))
        return risk_per_post

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 0, seed: int = 0, refit: bool = True,
    ) -> DoseResponseResult:
        """Posterior credible-interval dose-response (no bootstrap)."""
        if refit and self._posterior is None:
            self.fit(cohort)
        rng = np.random.default_rng(seed)
        K = len(target_bins)
        S = min(self.config.n_posterior_subset, self._posterior["beta"].shape[0])
        risk_mat = np.zeros((K, S), dtype=np.float64)
        for ki, k in enumerate(target_bins):
            risk_mat[ki] = self._counterfactual_per_bin(cohort, k, rng)
        return DoseResponseResult(
            bins=list(target_bins),
            bin_centers_J_min=bin_centers_J_min(cohort),
            risk_mean=risk_mat.mean(axis=1),
            risk_ci_low=np.quantile(risk_mat, 0.025, axis=1),
            risk_ci_high=np.quantile(risk_mat, 0.975, axis=1),
            risk_raw=risk_mat,
            method_name=self.method_name,
        )
