"""Abstract base for latent state-space models with treatment effects."""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseLatentSSM(torch.nn.Module, ABC):
    """Latent SSM interface decoupling model spec from inference and g-formula."""

    @abstractmethod
    def transition(
        self, Z_prev: Tensor, drivers: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (E[Z_t | Z_{t-1}, D_t], innovation variance)."""

    @abstractmethod
    def outcome_logit(self, Z: Tensor, covariates: Tensor) -> Tensor:
        """Return logit P(Y_t = 1 | Z_t, covariates)."""

    @property
    @abstractmethod
    def beta_Z(self) -> Tensor:
        """Coefficient of Z in the outcome logit."""

    @abstractmethod
    def initial_state(
        self, n_samples: int, device: torch.device,
        C_static: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Prior mean and variance of Z_1, optionally conditioned on baseline C."""
