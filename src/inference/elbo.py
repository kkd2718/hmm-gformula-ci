"""ELBO for the Linear-Gaussian SSM with Bernoulli outcome."""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.linear_gaussian_ssm import LinearGaussianSSM
from ..models.variational_posterior import StructuredGaussianMarkovPosterior


@dataclass
class ELBOResult:
    """Per-batch ELBO and component breakdown."""
    elbo: Tensor
    expected_log_lik: Tensor
    kl_dynamics: Tensor
    n_observations: int


def gaussian_kl(
    mu_q: Tensor, var_q: Tensor, mu_p: Tensor, var_p: Tensor, eps: float = 1e-8,
) -> Tensor:
    """Closed-form KL(q || p) for univariate Gaussians."""
    var_p = var_p + eps
    var_q = var_q + eps
    return 0.5 * (
        torch.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
    )


def compute_elbo(
    model: LinearGaussianSSM,
    posterior: StructuredGaussianMarkovPosterior,
    Y: Tensor,
    drivers: Tensor,
    covariates: Tensor,
    at_risk: Tensor,
    C_static: Tensor | None = None,
    n_samples: int = 4,
    smoothness_lambda: float = 0.0,
) -> ELBOResult:
    """ELBO = E_q[log p(Y|Z)] - KL(q(Z) || p(Z|C)) - smoothness penalty.

    at_risk masking applied to BOTH log-likelihood and KL chain so that
    post-event (absorbing) timesteps contribute neither term.
    """
    N, T, _ = Y.shape
    Z_samples, q_means, q_log_vars = posterior.sample_trajectory(
        drivers=drivers, covariates=covariates, Y=Y, n_samples=n_samples
    )

    log_lik = Y.new_zeros(())
    for s in range(n_samples):
        Z_s = Z_samples[s]
        Z_flat = Z_s.reshape(N * T, 1)
        cov_flat = covariates.reshape(N * T, -1)
        logit = model.outcome_logit(Z_flat, cov_flat).reshape(N, T, 1)
        log_p1 = -F.softplus(-logit)
        log_p0 = -F.softplus(logit)
        log_lik = log_lik + ((Y * log_p1 + (1 - Y) * log_p0) * at_risk).sum()
    expected_log_lik = log_lik / n_samples

    mu_q_init, var_q_init = posterior.initial_params(
        D_0=drivers[:, 0, :], W_0=covariates[:, 0, :], Y_0=Y[:, 0, :]
    )
    mu_p_init, var_p_init = model.initial_state(N, Y.device, C_static=C_static)
    kl_total = (
        gaussian_kl(mu_q_init, var_q_init, mu_p_init, var_p_init) * at_risk[:, 0, :]
    ).sum()

    for t in range(1, T):
        kl_t_mc = Y.new_zeros(())
        m_t = at_risk[:, t, :]
        for s in range(n_samples):
            Z_prev = Z_samples[s, :, t - 1, :]
            mu_q_t = q_means[s, :, t, :]
            var_q_t = torch.exp(q_log_vars[s, :, t, :])
            mu_p_t, var_p_t = model.transition(Z_prev, drivers[:, t, :])
            kl_t_mc = kl_t_mc + (
                gaussian_kl(mu_q_t, var_q_t, mu_p_t, var_p_t) * m_t
            ).sum()
        kl_total = kl_total + kl_t_mc / n_samples

    pen = smoothness_lambda * model.smoothness_penalty()
    elbo = expected_log_lik - kl_total - pen
    return ELBOResult(
        elbo=elbo,
        expected_log_lik=expected_log_lik.detach(),
        kl_dynamics=kl_total.detach(),
        n_observations=int(at_risk.sum().detach().cpu()),
    )
