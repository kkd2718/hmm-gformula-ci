"""Monte Carlo g-formula simulation for VEM-SSM (NICE algorithm).

Per Robins (1986) NICE form, all three time-varying quantities are simulated
forward under the counterfactual intervention A_t = a:
    Z_t  ~ p(Z_t | Z_{t-1}, A_{t-1}=a, L_{t-1}, V)       (option B uses A_lag)
    L_t  ~ p(L_t | Z_t, A_{t-1}=a, L_{t-1}, V)
    Y_t  ~ Bernoulli( logit( ... | Z_t, A_t=a, L_t, V, t ) )
After the per-subject event the Z trajectory is frozen (absorbing convention)
and survival probability decays multiplicatively.
"""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..models.linear_gaussian_ssm import LinearGaussianSSM


@torch.no_grad()
def simulate_g_formula(
    model: LinearGaussianSSM,
    A_baseline: Tensor,                 # (M, T, K) — baseline A_t for all t (overridden by intervene_bin)
    L_baseline_t0: Tensor,              # (M, p_dyn) — baseline L_0 from observed empirical
    V: Tensor,                          # (M, p_static)
    t_norm: Tensor,                     # (M, T, 1)
    intervene_bin: int,
    n_bins: int,
    sample_Z: bool = True,
    sample_L: bool = True,
    seed: Optional[int] = None,
) -> dict:
    """Forward-simulate (Z, L, Y) under counterfactual A_t = intervene_bin.

    Uses the empirical baseline (V, L_0) drawn by the caller; everything else
    is simulated from the fitted SSM equations.
    """
    gen = (
        torch.Generator(device=A_baseline.device).manual_seed(seed)
        if seed is not None else None
    )
    M, T, _ = A_baseline.shape
    device = A_baseline.device
    p_dyn = L_baseline_t0.shape[-1]

    one_hot = torch.zeros(n_bins, device=device)
    one_hot[intervene_bin] = 1.0
    A_cf = one_hot.unsqueeze(0).unsqueeze(0).expand(M, T, n_bins).clone()

    # Initial state
    Z_mean, Z_var = model.initial_state(V, M, device)
    if sample_Z:
        Z = Z_mean + torch.sqrt(Z_var) * torch.randn(
            Z_mean.shape, generator=gen, device=device
        )
    else:
        Z = Z_mean
    L_t = L_baseline_t0.clone()

    Z_path, L_path, prob_path, cum_path = [], [], [], []
    survived = torch.ones(M, 1, device=device)
    cumulative = torch.zeros(M, 1, device=device)
    zeros_A = A_cf.new_zeros((M, n_bins))
    zeros_L = L_t.new_zeros((M, p_dyn)) if p_dyn > 0 else L_t.new_zeros((M, 0))

    for t in range(T):
        if t > 0:
            # Z_t ~ p(Z_t | Z_{t-1}, A_{t-1}=a, L_{t-1}, V)
            A_lag = A_cf[:, t - 1, :]
            L_lag = L_t  # value of L_{t-1} from the previous iteration
            Z_next_mean, Z_var_add = model.transition(Z, A_lag, L_lag, V)
            if sample_Z:
                Z_new = Z_next_mean + torch.sqrt(Z_var_add) * torch.randn(
                    Z_next_mean.shape, generator=gen, device=device
                )
            else:
                Z_new = Z_next_mean
            # Freeze Z for absorbed (event-experienced) subjects
            Z = survived * Z_new + (1.0 - survived) * Z
            # L_t ~ p(L_t | Z_t, A_{t-1}=a, L_{t-1}, V)
            if p_dyn > 0:
                L_mean, L_var = model.l_emission(Z, A_lag, L_lag, V)
                if sample_L:
                    L_t = L_mean + torch.sqrt(L_var) * torch.randn(
                        L_mean.shape, generator=gen, device=device
                    )
                else:
                    L_t = L_mean
        else:
            # t = 0: Z_0 already drawn; L_0 from observed baseline; no A_lag/L_lag
            pass

        logit = model.outcome_logit(
            Z, A_cf[:, t, :], L_t, V, t_norm[:, t, :],
        )
        p_Y = torch.sigmoid(logit)
        new_event = survived * p_Y
        cumulative = cumulative + new_event
        survived = survived * (1.0 - p_Y)

        Z_path.append(Z)
        L_path.append(L_t)
        prob_path.append(p_Y)
        cum_path.append(cumulative.clone())

    return {
        "Z": torch.stack(Z_path, dim=1),
        "L": torch.stack(L_path, dim=1),
        "prob_Y": torch.stack(prob_path, dim=1),
        "cumulative_risk": torch.stack(cum_path, dim=1),
        "marginal_risk_T": cumulative.mean(),
    }
