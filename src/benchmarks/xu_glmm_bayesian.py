"""Xu 2024 GLMM — Bayesian MCMC implementation (numpyro / NUTS).

Faithful replication of Xu et al. (2024) Biometrics 80(3):ujae100. The
authors use full Bayesian g-computation with MCMC; the prior earlier
implementation in this codebase used a frequentist Laplace approximation
with a method-of-moments sigma^2 update which collapses sigma_b to the
floor on this cohort.

Generative model
----------------
    beta_0, beta_A, eta, xi  ~  N(0, 5^2)         weakly informative fixed-effect priors
    sigma_b                  ~  HalfCauchy(0, 2.5)   Gelman 2006 standard
    b_i | sigma_b            ~  N(0, sigma_b^2)
    logit P(Y_it = 1)        =  beta_0 + beta_A^T A_it + eta^T L_it + xi^T V_i + b_i

Counterfactual (matches Xu 2024 procedure: observed L plug-in, marginalize over b)
---------------------------------------------------------------------------------
For each posterior draw s = 1, ..., S:
    For each subject i:
        Use posterior draw (beta^s, eta^s, xi^s, sigma_b^s)
        Sample b_i^(s,m) ~ N(0, (sigma_b^s)^2),  m = 1, ..., M_b
        For each target bin k:
            For t = 0, ..., T-1:
                logit_t = ... + b_i^(s,m)            (no forward L sim; Xu MSM-style)
                p_t = sigmoid(logit_t)
                cum += survived * p_t
                survived *= (1 - p_t)
            Record I_i^(s, m, k)
Posterior credible interval on R(a_k) = mean over (i, m, s) of I_i^(s, m, k).

Notes
-----
- No cluster bootstrap (posterior CI replaces it; consistent with Xu 2024).
- Counterfactual b draws nested inside posterior draws to integrate over both.
- Convergence diagnostics: R-hat, ESS via numpyro.diagnostics.
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


def _xu_bayesian_model(X: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray,
                       group_idx: jnp.ndarray, n_groups: int) -> None:
    """numpyro model for Xu 2024 GLMM.

    X : (N_obs, p)  pooled design matrix (rows = subject-time)
    y : (N_obs,)    binary outcome
    mask : (N_obs,) at-risk indicator (0/1)
    group_idx : (N_obs,) integer subject index in [0, n_groups)
    """
    p = X.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 5.0))
    sigma_b = numpyro.sample("sigma_b", dist.HalfCauchy(2.5))
    # Non-centered parameterization for hierarchical prior (better mixing)
    z = numpyro.sample("z_b", dist.Normal(jnp.zeros(n_groups), 1.0))
    b = numpyro.deterministic("b", sigma_b * z)
    logit = X @ beta + b[group_idx]
    # Mask non-at-risk observations from likelihood
    log_p = mask * dist.Bernoulli(logits=logit).log_prob(y)
    numpyro.factor("loglik", log_p.sum())


@dataclass
class XuBayesianConfig:
    """Sampler / counterfactual hyperparameters."""
    n_warmup: int = 1000
    n_samples: int = 1000
    n_chains: int = 4
    chain_method: str = "parallel"
    target_accept: float = 0.9
    n_b_draws: int = 50              # MC draws over b per posterior sample
    n_posterior_subset: int = 200    # posterior draws used for counterfactual
    seed: int = 0


class XuGLMMBayesian(BenchmarkMethod):
    """Bayesian replication of Xu 2024 GLMM via numpyro NUTS.

    Replaces the broken Laplace + MoM implementation; faithful to the
    original Bayesian g-computation procedure.
    """

    method_name = "xu_glmm_bayesian"

    def __init__(self, config: XuBayesianConfig | None = None) -> None:
        self.config = config or XuBayesianConfig()
        self._posterior: dict | None = None
        self._n_groups: int = 0

    # ------------------------------------------------------------------
    # Design construction (same shape as legacy XuGLMM)
    # ------------------------------------------------------------------
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
        # Subject-level groups by integer index
        _, inv = np.unique(cohort.subject_ids, return_inverse=True)
        group_idx = np.repeat(inv, T)
        return X, y.astype(np.float64), m.astype(np.float64), group_idx

    # ------------------------------------------------------------------
    # NUTS fit
    # ------------------------------------------------------------------
    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        X, y, m, group_idx = self._build_design(cohort)
        n_groups = int(group_idx.max() + 1)
        self._n_groups = n_groups

        # MCMC
        kernel = NUTS(_xu_bayesian_model, target_accept_prob=self.config.target_accept)
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
            X=jnp.asarray(X), y=jnp.asarray(y), mask=jnp.asarray(m),
            group_idx=jnp.asarray(group_idx), n_groups=n_groups,
        )
        # Posterior samples (chain-flattened) for downstream MC
        samples_flat = mcmc.get_samples()
        self._posterior = {
            "beta": np.asarray(samples_flat["beta"]),     # (S, p)
            "sigma_b": np.asarray(samples_flat["sigma_b"]),  # (S,)
        }
        # Convergence diagnostics need chain dim — get_samples(group_by_chain=True)
        from numpyro.diagnostics import summary
        try:
            samples_chains = mcmc.get_samples(group_by_chain=True)
            diag = summary(samples_chains, prob=0.95)
            sig_diag = diag.get("sigma_b", {})
            r_hat = float(sig_diag.get("r_hat", float("nan")))
            n_eff = float(sig_diag.get("n_eff", float("nan")))
        except Exception:
            r_hat, n_eff = float("nan"), float("nan")
        print(
            f"  [Xu Bayesian fit] sigma_b posterior mean = "
            f"{self._posterior['sigma_b'].mean():.4f}, "
            f"R-hat = {r_hat:.3f}, ESS = {n_eff:.0f}"
        )

    # ------------------------------------------------------------------
    # Counterfactual via posterior + b-marginalization
    # ------------------------------------------------------------------
    def _counterfactual_per_bin(
        self, cohort: ARDSCohort, k: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-mean and credible-interval risk under A=k.

        Returns
        -------
        risk_per_posterior : (S_subset,)  population-mean risk per posterior draw,
                             integrated over b ~ N(0, sigma_b^2_s).
        """
        X, _, _, _ = self._build_design(cohort, override_bin=k)
        N, T = cohort.Y.shape[0], cohort.Y.shape[1]
        S_total = self._posterior["beta"].shape[0]
        S = min(self.config.n_posterior_subset, S_total)
        # Subsample posterior draws (uniformly)
        idx = rng.choice(S_total, size=S, replace=False)
        beta_post = self._posterior["beta"][idx]            # (S, p)
        sigma_b_post = self._posterior["sigma_b"][idx]      # (S,)

        risk_per_post = np.zeros(S, dtype=np.float64)
        # Loop over posterior draws — vectorize over subjects + b draws
        for s in range(S):
            beta = beta_post[s]
            sigma_b = sigma_b_post[s]
            eta_no_b = (X @ beta).reshape(N, T)             # (N, T)
            risks_b = []
            for _ in range(self.config.n_b_draws):
                b_per_subject = rng.normal(0.0, sigma_b, size=N)
                eta = eta_no_b + b_per_subject[:, None]     # (N, T)
                p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
                survived = np.ones(N, dtype=np.float64)
                cum = np.zeros(N, dtype=np.float64)
                for t in range(T):
                    cum = cum + survived * p[:, t]
                    survived = survived * (1.0 - p[:, t])
                risks_b.append(cum.mean())
            risk_per_post[s] = float(np.mean(risks_b))
        return risk_per_post

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 0, seed: int = 0, refit: bool = True,
    ) -> DoseResponseResult:
        """Posterior credible-interval dose-response (no bootstrap; Xu 2024 standard).

        n_bootstrap is accepted for API parity but ignored.
        """
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
