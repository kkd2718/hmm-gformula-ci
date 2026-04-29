"""Structured Gaussian Markov variational posterior q_phi(Z_{0:T-1})."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class QConfig:
    """Hyperparameters for StructuredGaussianMarkovPosterior."""
    n_drivers: int
    n_covariates: int
    init_log_sigma: float = -1.0
    use_Y_lag_in_conditional: bool = True
    use_Y_in_init: bool = False


class StructuredGaussianMarkovPosterior(nn.Module):
    """Forward-filter Gaussian Markov posterior with lagged-outcome conditioning.

    Conditional (t >= 1):
        q(Z_t | Z_{t-1}, A_t, L_t, C, Y_{t-1})
            = N(psi_q*Z_{t-1} + gamma_q^T D_t + alpha_q^T W_t + alpha_Ylag*Y_{t-1},
                tilde_sigma^2)
    Initial (t = 0): structural-only by default
        q(Z_0 | A_0, L_0, C) = N(mu_init + gamma_q^T D_0 + alpha_q^T W_0, sigma_init^2)
    Lagged Y_{t-1} preserves causal identifiability of beta_Z (Cappe Moulines Ryden 2005 sec 11.2).
    """

    def __init__(self, config: QConfig) -> None:
        super().__init__()
        self.config = config
        self.tilde_psi = nn.Parameter(torch.tensor([0.5]))
        self.gamma_q = nn.Linear(config.n_drivers, 1, bias=False)
        nn.init.zeros_(self.gamma_q.weight)
        self.alpha_q = nn.Linear(config.n_covariates, 1, bias=False)
        nn.init.zeros_(self.alpha_q.weight)
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

    def conditional_params(
        self, Z_prev: Tensor, D_t: Tensor, W_t: Tensor, Y_prev: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mu = self.tilde_psi * Z_prev + self.gamma_q(D_t) + self.alpha_q(W_t)
        if self.config.use_Y_lag_in_conditional:
            mu = mu + self.alpha_Y_lag * Y_prev
        var = (self.tilde_sigma ** 2).expand_as(mu)
        return mu, var

    def initial_params(
        self, n_samples: Optional[int] = None, device: Optional[torch.device] = None,
        D_0: Optional[Tensor] = None, W_0: Optional[Tensor] = None,
        Y_0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if D_0 is not None and W_0 is not None:
            mu = self.mu_init + self.gamma_q(D_0) + self.alpha_q(W_0)
            if self.config.use_Y_in_init and Y_0 is not None:
                mu = mu + self.alpha_Y * Y_0
            return mu, (self.sigma_init ** 2).expand_as(mu)
        if n_samples is None or device is None:
            raise ValueError("initial_params: provide (D_0,W_0) or (n_samples,device).")
        mu = self.mu_init.expand(n_samples, 1).to(device)
        return mu, (self.sigma_init ** 2).expand(n_samples, 1).to(device)

    def sample_trajectory(
        self, drivers: Tensor, covariates: Tensor, Y: Tensor,
        n_samples: int = 1, generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        N, T, _ = Y.shape
        device = Y.device
        all_Z, all_mu, all_lv = [], [], []
        for _ in range(n_samples):
            mu_0, var_0 = self.initial_params(
                D_0=drivers[:, 0, :], W_0=covariates[:, 0, :], Y_0=Y[:, 0, :],
            )
            Z_t = mu_0 + torch.sqrt(var_0) * torch.randn(
                N, 1, device=device, generator=generator
            )
            traj_Z, traj_mu, traj_lv = [Z_t], [mu_0], [torch.log(var_0)]
            for t in range(1, T):
                mu_t, var_t = self.conditional_params(
                    Z_prev=Z_t, D_t=drivers[:, t, :], W_t=covariates[:, t, :],
                    Y_prev=Y[:, t - 1, :],
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
