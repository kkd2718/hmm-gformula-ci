"""Structured Gaussian Markov variational posterior q_phi(Z_{0:T-1}).

Forward filter form (Cappe Moulines Ryden 2005 sec 11.2):
    q(Z_0   | A_0, L_0, V)              for t = 0
    q(Z_t   | Z_{t-1}, A_t, L_t, V, Y_{t-1})    for t >= 1

Lagged Y_{t-1} preserves causal identifiability of beta_Z (current Y_t is
NOT used in the encoder of Z_t — it would let Z_t encode the outcome it
is meant to confound).

Inputs to the encoder: A (one-hot), L (continuous), V (continuous).
The encoder is independent of the generative model's choice of option A vs B
(both options share the same set of observed inputs).
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class QConfig:
    """Hyperparameters for StructuredGaussianMarkovPosterior."""
    n_bins: int
    n_dyn_covariates: int
    n_static_covariates: int
    init_log_sigma: float = -1.0
    use_Y_lag_in_conditional: bool = True
    use_Y_in_init: bool = False


class StructuredGaussianMarkovPosterior(nn.Module):
    """Markov forward filter with linear-Gaussian conditionals.

    q(Z_t | history_t) = N(mu_q(history_t), sigma_q^2)
    where mu_q is linear in (Z_{t-1}, A, L, V, Y_lag).
    """

    def __init__(self, config: QConfig) -> None:
        super().__init__()
        self.config = config
        K, p_dyn, p_stat = (
            config.n_bins, config.n_dyn_covariates, config.n_static_covariates,
        )
        self.tilde_psi = nn.Parameter(torch.tensor([0.5]))
        self.gamma_A_q = (
            nn.Parameter(torch.zeros(K)) if K > 0 else None
        )
        self.gamma_L_q = (
            nn.Linear(p_dyn, 1, bias=False) if p_dyn > 0 else None
        )
        if self.gamma_L_q is not None:
            nn.init.zeros_(self.gamma_L_q.weight)
        self.gamma_V_q = (
            nn.Linear(p_stat, 1, bias=False) if p_stat > 0 else None
        )
        if self.gamma_V_q is not None:
            nn.init.zeros_(self.gamma_V_q.weight)
        self.alpha_Y_lag = nn.Parameter(torch.tensor([0.0]))
        self.alpha_Y = nn.Parameter(torch.tensor([0.0]))
        self.log_tilde_sigma = nn.Parameter(torch.tensor([config.init_log_sigma]))
        self.mu_init = nn.Parameter(torch.tensor([0.0]))
        self.log_sigma_init = nn.Parameter(torch.tensor([0.0]))

    @property
    def tilde_sigma(self) -> Tensor:
        return torch.exp(self.log_tilde_sigma)

    @property
    def sigma_init(self) -> Tensor:
        return torch.exp(self.log_sigma_init)

    def _encoder_mean(
        self, Z_prev: Tensor | None, A: Tensor, L: Tensor, V: Tensor, Y_lag: Tensor | None,
    ) -> Tensor:
        device = A.device if A is not None else (V.device if V is not None else None)
        N = (A.shape[0] if A is not None else (V.shape[0] if V is not None else 0))
        mu = (
            self.tilde_psi * Z_prev
            if Z_prev is not None else
            self.mu_init.expand(N, 1).to(device)
        )
        if self.gamma_A_q is not None and A is not None:
            mu = mu + (A * self.gamma_A_q).sum(-1, keepdim=True)
        if self.gamma_L_q is not None and L is not None:
            mu = mu + self.gamma_L_q(L)
        if self.gamma_V_q is not None and V is not None:
            mu = mu + self.gamma_V_q(V)
        if Y_lag is not None and self.config.use_Y_lag_in_conditional:
            mu = mu + self.alpha_Y_lag * Y_lag
        return mu

    def conditional_params(
        self, Z_prev: Tensor, A: Tensor, L: Tensor, V: Tensor, Y_lag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mu = self._encoder_mean(Z_prev, A, L, V, Y_lag)
        var = (self.tilde_sigma ** 2).expand_as(mu)
        return mu, var

    def initial_params(
        self, A_0: Tensor, L_0: Tensor, V: Tensor, Y_0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        mu = self._encoder_mean(None, A_0, L_0, V, None)
        if self.config.use_Y_in_init and Y_0 is not None:
            mu = mu + self.alpha_Y * Y_0
        return mu, (self.sigma_init ** 2).expand_as(mu)

    def sample_trajectory(
        self, A: Tensor, L: Tensor, V: Tensor, Y: Tensor,
        n_samples: int = 1, generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (Z_samples, q_means, q_log_vars) of shape (S, N, T, 1).

        A, L: (N, T, K) and (N, T, p_dyn). V: (N, p_static). Y: (N, T, 1).
        """
        N, T, _ = Y.shape
        device = Y.device
        all_Z, all_mu, all_lv = [], [], []
        for _ in range(n_samples):
            mu_0, var_0 = self.initial_params(
                A_0=A[:, 0, :], L_0=L[:, 0, :], V=V, Y_0=Y[:, 0, :],
            )
            Z_t = mu_0 + torch.sqrt(var_0) * torch.randn(
                N, 1, device=device, generator=generator
            )
            traj_Z, traj_mu, traj_lv = [Z_t], [mu_0], [torch.log(var_0)]
            for t in range(1, T):
                mu_t, var_t = self.conditional_params(
                    Z_prev=Z_t, A=A[:, t, :], L=L[:, t, :], V=V,
                    Y_lag=Y[:, t - 1, :],
                )
                Z_t = mu_t + torch.sqrt(var_t) * torch.randn(
                    N, 1, device=device, generator=generator
                )
                traj_Z.append(Z_t)
                traj_mu.append(mu_t)
                traj_lv.append(torch.log(var_t))
            all_Z.append(torch.stack(traj_Z, dim=1))
            all_mu.append(torch.stack(traj_mu, dim=1))
            all_lv.append(torch.stack(traj_lv, dim=1))
        return (
            torch.stack(all_Z, dim=0),
            torch.stack(all_mu, dim=0),
            torch.stack(all_lv, dim=0),
        )
