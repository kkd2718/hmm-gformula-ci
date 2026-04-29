"""Linear-Gaussian SSM with Bernoulli outcome (proposed model)."""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseLatentSSM


@dataclass
class SSMConfig:
    """Hyperparameters for LinearGaussianSSM."""
    n_bins: int
    n_dyn_covariates: int
    n_static_covariates: int
    fit_time_effect: bool = True
    init_psi: float = 0.5
    init_log_sigma_Z: float = -2.0
    init_beta_0: float = -4.5
    init_beta_Z: float = 0.3


class LinearGaussianSSM(BaseLatentSSM):
    """Continuous latent Z_t with linear-Gaussian dynamics and Bernoulli outcome.

    Z_t = psi*Z_{t-1} + gamma_A^T A_t + gamma_dyn^T L_t + gamma_static^T C + N(0, sigma_Z^2)
    logit P(Y_t=1) = beta_0 + beta_Z*Z_t + beta_A^T A_t + beta_dyn^T L_t
                     + beta_static^T C + beta_time*(t/T_max)
    Baseline-conditioned prior: p(Z_1 | C) = N(gamma_init^T C, sigma_init0^2).
    """

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.config = config

        self.psi = nn.Parameter(torch.tensor([config.init_psi]))
        self.gamma_A = nn.Parameter(torch.zeros(config.n_bins))
        self.gamma_dyn = (
            nn.Linear(config.n_dyn_covariates, 1, bias=False)
            if config.n_dyn_covariates > 0 else None
        )
        if self.gamma_dyn is not None:
            nn.init.normal_(self.gamma_dyn.weight, mean=0.0, std=0.05)
        self.gamma_static = (
            nn.Linear(config.n_static_covariates, 1, bias=False)
            if config.n_static_covariates > 0 else None
        )
        if self.gamma_static is not None:
            nn.init.normal_(self.gamma_static.weight, mean=0.0, std=0.05)
        self.log_sigma_Z = nn.Parameter(torch.tensor([config.init_log_sigma_Z]))

        self.beta_0_param = nn.Parameter(torch.tensor([config.init_beta_0]))
        self._beta_Z = nn.Parameter(torch.tensor([config.init_beta_Z]))
        self.beta_A = nn.Parameter(torch.zeros(config.n_bins))
        self.beta_dyn = (
            nn.Linear(config.n_dyn_covariates, 1, bias=False)
            if config.n_dyn_covariates > 0 else None
        )
        if self.beta_dyn is not None:
            nn.init.normal_(self.beta_dyn.weight, mean=0.0, std=0.1)
        self.beta_static = (
            nn.Linear(config.n_static_covariates, 1, bias=False)
            if config.n_static_covariates > 0 else None
        )
        if self.beta_static is not None:
            nn.init.normal_(self.beta_static.weight, mean=0.0, std=0.1)
        self.beta_time = (
            nn.Parameter(torch.tensor([0.0])) if config.fit_time_effect else None
        )

        self.gamma_init = (
            nn.Linear(config.n_static_covariates, 1, bias=False)
            if config.n_static_covariates > 0 else None
        )
        if self.gamma_init is not None:
            nn.init.zeros_(self.gamma_init.weight)
        self.log_sigma_init0 = nn.Parameter(torch.tensor([0.0]))

    @property
    def sigma_Z(self) -> Tensor:
        return torch.exp(self.log_sigma_Z)

    @property
    def beta_Z(self) -> Tensor:
        return self._beta_Z

    def transition(self, Z_prev: Tensor, drivers: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        K, d, s = cfg.n_bins, cfg.n_dyn_covariates, cfg.n_static_covariates
        A_bin = drivers[:, :K]
        L_t = drivers[:, K:K + d] if d > 0 else None
        C = drivers[:, K + d:K + d + s] if s > 0 else None
        Z_mean = self.psi * Z_prev + (A_bin * self.gamma_A).sum(-1, keepdim=True)
        if L_t is not None and self.gamma_dyn is not None:
            Z_mean = Z_mean + self.gamma_dyn(L_t)
        if C is not None and self.gamma_static is not None:
            Z_mean = Z_mean + self.gamma_static(C)
        Z_var_add = (self.sigma_Z ** 2).expand_as(Z_mean)
        return Z_mean, Z_var_add

    def outcome_logit(self, Z: Tensor, covariates: Tensor) -> Tensor:
        cfg = self.config
        K, d, s = cfg.n_bins, cfg.n_dyn_covariates, cfg.n_static_covariates
        A_bin = covariates[:, :K]
        L_t = covariates[:, K:K + d] if d > 0 else None
        C = covariates[:, K + d:K + d + s] if s > 0 else None
        t_norm = covariates[:, K + d + s:K + d + s + 1] if cfg.fit_time_effect else None
        logit = self.beta_0_param + self._beta_Z * Z
        logit = logit + (A_bin * self.beta_A).sum(-1, keepdim=True)
        if L_t is not None and self.beta_dyn is not None:
            logit = logit + self.beta_dyn(L_t)
        if C is not None and self.beta_static is not None:
            logit = logit + self.beta_static(C)
        if t_norm is not None and self.beta_time is not None:
            logit = logit + self.beta_time * t_norm
        return logit

    def initial_state(
        self, n_samples: int, device: torch.device,
        C_static: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        var0 = (torch.exp(self.log_sigma_init0) ** 2)
        if C_static is not None and self.gamma_init is not None:
            mu0 = self.gamma_init(C_static)
            return mu0, var0.expand_as(mu0)
        mu0 = torch.zeros(n_samples, 1, device=device)
        return mu0, var0.expand(n_samples, 1).to(device)

    def smoothness_penalty(self) -> Tensor:
        diffs = self.beta_A[1:] - self.beta_A[:-1]
        return (diffs ** 2).sum()
