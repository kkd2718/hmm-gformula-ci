"""Abstract base for latent state-space models with generative L emission."""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseLatentSSM(torch.nn.Module, ABC):
    """Latent SSM interface for VEM-SSM g-formula.

    Three model components:
      transition(Z_prev, A_lag, L_lag, V) -> p(Z_t | history): latent dynamics
      l_emission(Z, A_lag, L_lag, V)      -> p(L_t | Z_t, history): observed lab generation
      outcome_logit(Z, A, L, V, t)        -> logit P(Y_t = 1): hazard model

    Concrete subclasses (LinearGaussianSSM) realize each component.
    """

    @abstractmethod
    def transition(
        self,
        Z_prev: Tensor,        # (N, 1)
        A_lag: Tensor,         # (N, K) one-hot of A_{t-1}
        L_lag: Tensor,         # (N, p_dyn) of L_{t-1}
        V: Tensor,             # (N, p_static)
    ) -> tuple[Tensor, Tensor]:
        """Return (E[Z_t | history], innovation variance σ_Z²) for the latent SSM."""

    @abstractmethod
    def l_emission(
        self,
        Z: Tensor,             # (N, 1)
        A_lag: Tensor,         # (N, K)
        L_lag: Tensor,         # (N, p_dyn)
        V: Tensor,             # (N, p_static)
    ) -> tuple[Tensor, Tensor]:
        """Return per-dim mean and variance of p(L_t | Z_t, A_{t-1}, L_{t-1}, V)."""

    @abstractmethod
    def outcome_logit(
        self,
        Z: Tensor,             # (N, 1)
        A: Tensor,             # (N, K)
        L: Tensor,             # (N, p_dyn)
        V: Tensor,             # (N, p_static)
        t_norm: Tensor,        # (N, 1) in [0, 1]
    ) -> Tensor:
        """Return logit P(Y_t = 1 | Z_t, A_t, L_t, V, t)."""

    @property
    @abstractmethod
    def beta_Z(self) -> Tensor:
        """Coefficient of Z in the outcome logit."""

    @abstractmethod
    def initial_state(
        self,
        V: Tensor | None,      # (N, p_static)
        n_samples: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Prior mean and variance of Z_0, conditioned on V if provided."""
