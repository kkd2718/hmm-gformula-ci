"""ELBO for the Linear-Gaussian SSM with generative L emission and Bernoulli outcome.

Joint generative likelihood:
    log p(Y, L | Z) + log p(Z | Z_prev, history)
    = log p(Y_t | Z_t, A_t, L_t, V, t) + log p(L_t | Z_t, A_{t-1}, L_{t-1}, V)
      + log p(Z_t | Z_{t-1}, [A_{t-1}], L_{t-1}, V)

ELBO:
    L(theta, phi) = E_q[ log p(Y | Z, A, L, V) + log p(L | Z, A_lag, L_lag, V) ]
                    - sum_t KL(q(Z_t) || p(Z_t | Z_{t-1}, ...))
                    - lambda * smoothness penalty

Both expectations and KLs are masked by at_risk so that post-event timesteps
contribute neither term (absorbing convention).
"""
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
    expected_log_lik_y: Tensor
    expected_log_lik_l: Tensor
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


def gaussian_log_prob_diag(
    x: Tensor, mu: Tensor, var: Tensor, eps: float = 1e-8,
) -> Tensor:
    """log N(x; mu, diag(var)), summed over the last dim."""
    var = var + eps
    return -0.5 * (
        torch.log(2 * torch.pi * var) + (x - mu) ** 2 / var
    ).sum(dim=-1, keepdim=True)


def compute_elbo(
    model: LinearGaussianSSM,
    posterior: StructuredGaussianMarkovPosterior,
    Y: Tensor,                      # (N, T, 1)
    A: Tensor,                      # (N, T, K) one-hot
    L: Tensor,                      # (N, T, p_dyn)
    V: Tensor,                      # (N, p_static)
    at_risk: Tensor,                # (N, T, 1)
    t_norm: Tensor,                 # (N, T, 1) in [0, 1]
    n_samples: int = 4,
    smoothness_lambda: float = 0.0,
) -> ELBOResult:
    """Compute the ELBO and its components."""
    N, T, _ = Y.shape

    Z_samples, q_means, q_log_vars = posterior.sample_trajectory(
        A=A, L=L, V=V, Y=Y, n_samples=n_samples,
    )

    # ===== E_q[log p(Y_t | Z_t, A_t, L_t, V, t)] =====
    log_lik_y = Y.new_zeros(())
    for s in range(n_samples):
        Z_s = Z_samples[s]
        Z_flat = Z_s.reshape(N * T, 1)
        A_flat = A.reshape(N * T, -1)
        L_flat = L.reshape(N * T, -1)
        V_rep = V.unsqueeze(1).expand(N, T, V.shape[-1]).reshape(N * T, -1)
        t_flat = t_norm.reshape(N * T, 1)
        logit = model.outcome_logit(
            Z_flat, A_flat, L_flat, V_rep, t_flat,
        ).reshape(N, T, 1)
        log_p1 = -F.softplus(-logit)
        log_p0 = -F.softplus(logit)
        log_lik_y = log_lik_y + ((Y * log_p1 + (1 - Y) * log_p0) * at_risk).sum()
    expected_log_lik_y = log_lik_y / n_samples

    # ===== E_q[log p(L_t | Z_t, A_{t-1}, L_{t-1}, V)] =====
    # For t=0 we use l0_intercept + delta_V*V + delta_Z*Z_0 (no A_lag, L_lag).
    log_lik_l = Y.new_zeros(())
    p_dyn = L.shape[-1]
    if p_dyn > 0:
        zeros_A = A.new_zeros((N, A.shape[-1]))
        zeros_L = L.new_zeros((N, p_dyn))
        for s in range(n_samples):
            Z_s = Z_samples[s]
            for t in range(T):
                Z_t = Z_s[:, t, :]
                A_lag = zeros_A if t == 0 else A[:, t - 1, :]
                L_lag = zeros_L if t == 0 else L[:, t - 1, :]
                mu_L, var_L = model.l_emission(Z_t, A_lag, L_lag, V)
                lp = gaussian_log_prob_diag(L[:, t, :], mu_L, var_L)
                log_lik_l = log_lik_l + (lp * at_risk[:, t, :]).sum()
        expected_log_lik_l = log_lik_l / n_samples
    else:
        expected_log_lik_l = log_lik_l

    # ===== KL(q(Z_0) || p(Z_0 | V)) =====
    mu_q_init, var_q_init = posterior.initial_params(
        A_0=A[:, 0, :], L_0=L[:, 0, :], V=V, Y_0=Y[:, 0, :],
    )
    mu_p_init, var_p_init = model.initial_state(V, N, Y.device)
    kl_total = (
        gaussian_kl(mu_q_init, var_q_init, mu_p_init, var_p_init) * at_risk[:, 0, :]
    ).sum()

    # ===== KL chain for t = 1..T-1 (averaged across MC samples) =====
    zeros_A = A.new_zeros((N, A.shape[-1]))
    zeros_L = L.new_zeros((N, p_dyn)) if p_dyn > 0 else L.new_zeros((N, 0))
    for t in range(1, T):
        kl_t_mc = Y.new_zeros(())
        m_t = at_risk[:, t, :]
        for s in range(n_samples):
            Z_prev = Z_samples[s, :, t - 1, :]
            mu_q_t = q_means[s, :, t, :]
            var_q_t = torch.exp(q_log_vars[s, :, t, :])
            A_lag = A[:, t - 1, :]
            L_lag = L[:, t - 1, :] if p_dyn > 0 else zeros_L
            mu_p_t, var_p_t = model.transition(Z_prev, A_lag, L_lag, V)
            kl_t_mc = kl_t_mc + (
                gaussian_kl(mu_q_t, var_q_t, mu_p_t, var_p_t) * m_t
            ).sum()
        kl_total = kl_total + kl_t_mc / n_samples

    pen = smoothness_lambda * model.smoothness_penalty()
    elbo = expected_log_lik_y + expected_log_lik_l - kl_total - pen

    return ELBOResult(
        elbo=elbo,
        expected_log_lik_y=expected_log_lik_y.detach(),
        expected_log_lik_l=expected_log_lik_l.detach(),
        kl_dynamics=kl_total.detach(),
        n_observations=int(at_risk.sum().detach().cpu()),
    )
