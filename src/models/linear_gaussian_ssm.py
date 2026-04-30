"""Linear-Gaussian SSM with generative L emission and Bernoulli Y emission.

Model (option B = full / primary):
    Z_0 ~ N(γ_init·V, σ_init²)
    Z_t = ψ Z_{t-1} + γ_A·A_{t-1} + γ_L·L_{t-1} + γ_V·V + ε_Z       (1)
    L_t = α_L·L_{t-1} + δ_A·A_{t-1} + δ_V·V + δ_Z·Z_t + ε_L         (2)
    logit P(Y_t=1) = β_0 + β_Z Z_t + β_A·A_t + β_L·L_t + β_V·V + β_t·(t/T)   (3)

Option A drops γ_A·A_{t-1} from equation (1): Z_t becomes exogenous to treatment.
Both options share equations (2) and (3) with Standard NICE g-formula structure.

Generalization of Xu (2024):
    Setting ψ=1 and γ_A=γ_L=0 reduces (1) to Z_t = γ_V·V + ε_Z, i.e. a
    random intercept with variance σ_Z² that does not change with time —
    exactly Xu's b_i ~ N(0, σ_b²) up to reparameterization.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseLatentSSM


@dataclass
class SSMConfig:
    """Hyperparameters for LinearGaussianSSM (option A or B)."""
    n_bins: int
    n_dyn_covariates: int
    n_static_covariates: int
    z_depends_on_treatment_lag: bool = True   # True = option B; False = option A
    fit_time_effect: bool = True
    init_psi: float = 0.5
    init_log_sigma_Z: float = -2.0
    init_log_sigma_L: float = -1.0
    init_beta_0: float = -3.0
    init_beta_Z: float = 0.3


class LinearGaussianSSM(BaseLatentSSM):
    """Continuous latent Z with linear-Gaussian dynamics, L emission, and Bernoulli Y."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.config = config
        K, p_dyn, p_stat = (
            config.n_bins, config.n_dyn_covariates, config.n_static_covariates,
        )

        # ---------- Latent dynamics (eq. 1) ----------
        self.psi = nn.Parameter(torch.tensor([config.init_psi]))
        # γ_A: A_{t-1} -> Z_t (option B only; held at zero and frozen for option A)
        self.gamma_A = nn.Parameter(torch.zeros(K))
        if not config.z_depends_on_treatment_lag:
            self.gamma_A.requires_grad_(False)
        self.gamma_L = (
            nn.Linear(p_dyn, 1, bias=False) if p_dyn > 0 else None
        )
        if self.gamma_L is not None:
            nn.init.normal_(self.gamma_L.weight, mean=0.0, std=0.05)
        self.gamma_V_dyn = (
            nn.Linear(p_stat, 1, bias=False) if p_stat > 0 else None
        )
        if self.gamma_V_dyn is not None:
            nn.init.normal_(self.gamma_V_dyn.weight, mean=0.0, std=0.05)
        self.log_sigma_Z = nn.Parameter(torch.tensor([config.init_log_sigma_Z]))

        # ---------- Initial Z prior p(Z_0 | V) ----------
        self.gamma_init = (
            nn.Linear(p_stat, 1, bias=False) if p_stat > 0 else None
        )
        if self.gamma_init is not None:
            nn.init.zeros_(self.gamma_init.weight)
        self.log_sigma_init0 = nn.Parameter(torch.tensor([0.0]))

        # ---------- L emission (eq. 2): L_t = α_L L_{t-1} + δ_A A_{t-1} + δ_V V + δ_Z Z_t ----------
        # Per-dim diagonal AR coefficient on L_{t-1}
        self.alpha_L = (
            nn.Parameter(torch.zeros(p_dyn)) if p_dyn > 0 else None
        )
        if self.alpha_L is not None:
            nn.init.constant_(self.alpha_L, 0.5)   # mild AR(1) by default
        # δ_A: K -> p_dyn (treatment lag affects each L dim)
        self.delta_A = (
            nn.Linear(K, p_dyn, bias=False) if (K > 0 and p_dyn > 0) else None
        )
        if self.delta_A is not None:
            nn.init.normal_(self.delta_A.weight, mean=0.0, std=0.05)
        # δ_V: p_stat -> p_dyn
        self.delta_V = (
            nn.Linear(p_stat, p_dyn, bias=False) if (p_stat > 0 and p_dyn > 0) else None
        )
        if self.delta_V is not None:
            nn.init.normal_(self.delta_V.weight, mean=0.0, std=0.05)
        # δ_Z: scalar coefficient per L dim
        self.delta_Z = (
            nn.Parameter(torch.zeros(p_dyn)) if p_dyn > 0 else None
        )
        # Per-dim L emission noise (parameterize log to keep positive)
        self.log_sigma_L = (
            nn.Parameter(torch.full((p_dyn,), config.init_log_sigma_L))
            if p_dyn > 0 else None
        )
        # L_0 mean from V (and Z_0 via δ_Z later)
        self.l0_intercept = (
            nn.Parameter(torch.zeros(p_dyn)) if p_dyn > 0 else None
        )

        # ---------- Outcome (eq. 3) ----------
        self.beta_0_param = nn.Parameter(torch.tensor([config.init_beta_0]))
        self._beta_Z = nn.Parameter(torch.tensor([config.init_beta_Z]))
        self.beta_A = nn.Parameter(torch.zeros(K))
        self.beta_L = (
            nn.Linear(p_dyn, 1, bias=False) if p_dyn > 0 else None
        )
        if self.beta_L is not None:
            nn.init.normal_(self.beta_L.weight, mean=0.0, std=0.1)
        self.beta_V_out = (
            nn.Linear(p_stat, 1, bias=False) if p_stat > 0 else None
        )
        if self.beta_V_out is not None:
            nn.init.normal_(self.beta_V_out.weight, mean=0.0, std=0.1)
        self.beta_time = (
            nn.Parameter(torch.tensor([0.0])) if config.fit_time_effect else None
        )

    @property
    def sigma_Z(self) -> Tensor:
        return torch.exp(self.log_sigma_Z)

    @property
    def sigma_L(self) -> Tensor:
        return torch.exp(self.log_sigma_L) if self.log_sigma_L is not None else None

    @property
    def beta_Z(self) -> Tensor:
        return self._beta_Z

    # --------------------------------------------------------------------
    # Equation (1): Z_t | Z_{t-1}, A_{t-1}, L_{t-1}, V
    # --------------------------------------------------------------------
    def transition(
        self, Z_prev: Tensor, A_lag: Tensor, L_lag: Tensor, V: Tensor,
    ) -> tuple[Tensor, Tensor]:
        Z_mean = self.psi * Z_prev
        if self.config.z_depends_on_treatment_lag:
            Z_mean = Z_mean + (A_lag * self.gamma_A).sum(-1, keepdim=True)
        if self.gamma_L is not None and L_lag is not None:
            Z_mean = Z_mean + self.gamma_L(L_lag)
        if self.gamma_V_dyn is not None and V is not None:
            Z_mean = Z_mean + self.gamma_V_dyn(V)
        Z_var = (self.sigma_Z ** 2).expand_as(Z_mean)
        return Z_mean, Z_var

    # --------------------------------------------------------------------
    # Equation (2): L_t | Z_t, A_{t-1}, L_{t-1}, V
    # --------------------------------------------------------------------
    def l_emission(
        self, Z: Tensor, A_lag: Tensor, L_lag: Tensor, V: Tensor,
    ) -> tuple[Tensor, Tensor]:
        N = Z.shape[0]
        p_dyn = self.config.n_dyn_covariates
        if p_dyn == 0:
            empty = Z.new_zeros((N, 0))
            return empty, empty
        L_mean = self.l0_intercept.unsqueeze(0).expand(N, p_dyn).clone()
        if L_lag is not None:
            L_mean = L_mean + self.alpha_L.unsqueeze(0) * L_lag
        if self.delta_A is not None and A_lag is not None:
            L_mean = L_mean + self.delta_A(A_lag)
        if self.delta_V is not None and V is not None:
            L_mean = L_mean + self.delta_V(V)
        if self.delta_Z is not None:
            L_mean = L_mean + self.delta_Z.unsqueeze(0) * Z
        L_var = (self.sigma_L ** 2).unsqueeze(0).expand(N, p_dyn)
        return L_mean, L_var

    # --------------------------------------------------------------------
    # Equation (3): logit P(Y_t = 1 | Z_t, A_t, L_t, V, t)
    # --------------------------------------------------------------------
    def outcome_logit(
        self, Z: Tensor, A: Tensor, L: Tensor, V: Tensor, t_norm: Tensor,
    ) -> Tensor:
        logit = self.beta_0_param + self._beta_Z * Z
        logit = logit + (A * self.beta_A).sum(-1, keepdim=True)
        if self.beta_L is not None and L is not None:
            logit = logit + self.beta_L(L)
        if self.beta_V_out is not None and V is not None:
            logit = logit + self.beta_V_out(V)
        if self.beta_time is not None and t_norm is not None:
            logit = logit + self.beta_time * t_norm
        return logit

    def initial_state(
        self, V: Tensor | None, n_samples: int, device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        var0 = torch.exp(self.log_sigma_init0) ** 2
        if V is not None and self.gamma_init is not None:
            mu0 = self.gamma_init(V)
            return mu0, var0.expand_as(mu0)
        mu0 = torch.zeros(n_samples, 1, device=device)
        return mu0, var0.expand(n_samples, 1).to(device)

    # --------------------------------------------------------------------
    # Smoothness penalty on β_A (and γ_A if option B)
    # --------------------------------------------------------------------
    def smoothness_penalty(self) -> Tensor:
        diffs_b = self.beta_A[1:] - self.beta_A[:-1]
        pen = (diffs_b ** 2).sum()
        if self.config.z_depends_on_treatment_lag:
            diffs_g = self.gamma_A[1:] - self.gamma_A[:-1]
            pen = pen + (diffs_g ** 2).sum()
        return pen
