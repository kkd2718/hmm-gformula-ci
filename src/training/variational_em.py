"""Stochastic variational EM (Adam) for joint optimization of theta and phi."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from ..models.linear_gaussian_ssm import LinearGaussianSSM
from ..models.variational_posterior import StructuredGaussianMarkovPosterior
from ..inference.elbo import compute_elbo, ELBOResult


@dataclass
class TrainingConfig:
    """Hyperparameters for the ELBO-maximization loop."""
    n_epochs: int = 300
    learning_rate: float = 1e-2
    n_mc_samples: int = 4
    smoothness_lambda: float = 0.02
    grad_clip: float = 5.0
    print_every: int = 25
    early_stop_patience: int = 50
    early_stop_tol: float = 1e-3


@dataclass
class TrainingHistory:
    """ELBO trajectory and components recorded each epoch."""
    elbo: list[float]
    expected_log_lik_y: list[float]
    expected_log_lik_l: list[float]
    kl: list[float]


def train_vem(
    model: LinearGaussianSSM,
    posterior: StructuredGaussianMarkovPosterior,
    Y: Tensor,                  # (N, T, 1)
    A: Tensor,                  # (N, T, K)
    L: Tensor,                  # (N, T, p_dyn)
    V: Tensor,                  # (N, p_static)
    at_risk: Tensor,            # (N, T, 1)
    t_norm: Tensor,             # (N, T, 1)
    config: TrainingConfig,
    device: torch.device | None = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Maximize ELBO over (theta, phi) jointly with Adam (Beal 2003; Hoffman 2013)."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    posterior.to(device)
    Y = Y.to(device); A = A.to(device); L = L.to(device); V = V.to(device)
    at_risk = at_risk.to(device); t_norm = t_norm.to(device)

    params: Iterable[Tensor] = list(model.parameters()) + list(posterior.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    history = TrainingHistory(elbo=[], expected_log_lik_y=[], expected_log_lik_l=[], kl=[])
    best_elbo = -float("inf")
    epochs_no_improve = 0

    for epoch in range(config.n_epochs):
        optimizer.zero_grad()
        result: ELBOResult = compute_elbo(
            model=model, posterior=posterior,
            Y=Y, A=A, L=L, V=V, at_risk=at_risk, t_norm=t_norm,
            n_samples=config.n_mc_samples,
            smoothness_lambda=config.smoothness_lambda,
        )
        loss = -result.elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=config.grad_clip)
        optimizer.step()

        elbo_val = float(result.elbo.detach().cpu())
        history.elbo.append(elbo_val)
        history.expected_log_lik_y.append(float(result.expected_log_lik_y.cpu()))
        history.expected_log_lik_l.append(float(result.expected_log_lik_l.cpu()))
        history.kl.append(float(result.kl_dynamics.cpu()))

        if elbo_val > best_elbo + config.early_stop_tol:
            best_elbo = elbo_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 0 or (epoch + 1) % config.print_every == 0):
            n_obs = max(result.n_observations, 1)
            print(
                f"[epoch {epoch+1:4d}/{config.n_epochs}] "
                f"ELBO={elbo_val:.2f}  per-obs={elbo_val / n_obs:.4f}  "
                f"ll_Y={history.expected_log_lik_y[-1]:.1f}  ll_L={history.expected_log_lik_l[-1]:.1f}  "
                f"KL={history.kl[-1]:.1f}"
            )
        if epochs_no_improve >= config.early_stop_patience:
            if verbose:
                print(f"[early stop] epoch {epoch+1}.")
            break

    return history
