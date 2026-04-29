"""Monte Carlo g-formula simulation under a fixed intervention regime."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..models.base import BaseLatentSSM


@torch.no_grad()
def simulate_g_formula(
    model: BaseLatentSSM,
    bootstrap_drivers: Tensor,
    bootstrap_covariates: Tensor,
    intervene_bin: int,
    n_bins: int,
    bin_slice: slice,
    C_static: Optional[Tensor] = None,
    sample_Z: bool = True,
    seed: Optional[int] = None,
) -> dict:
    """Forward-simulate Z and Y under A_t = intervene_bin for all t.

    Returns dict with 'Z', 'prob_Y', 'cumulative_risk', 'marginal_risk_T'.
    Latent state is frozen after the per-subject event (absorbing convention).
    """
    gen = (
        torch.Generator(device=bootstrap_drivers.device).manual_seed(seed)
        if seed is not None else None
    )
    M, T, _ = bootstrap_drivers.shape
    device = bootstrap_drivers.device

    one_hot = torch.zeros(n_bins, device=device)
    one_hot[intervene_bin] = 1.0
    drivers = bootstrap_drivers.clone()
    covariates = bootstrap_covariates.clone()
    drivers[:, :, bin_slice] = one_hot
    covariates[:, :, bin_slice] = one_hot

    Z_mean, Z_var = model.initial_state(M, device, C_static=C_static)
    Z = (
        Z_mean + torch.sqrt(Z_var) * torch.randn(Z_mean.shape, generator=gen, device=device)
        if sample_Z else Z_mean
    )

    Z_path, prob_path, cum_path = [], [], []
    survived = torch.ones(M, 1, device=device)
    cumulative = torch.zeros(M, 1, device=device)

    for t in range(T):
        Z_next_mean, Z_var_add = model.transition(Z, drivers[:, t, :])
        Z_new = (
            Z_next_mean + torch.sqrt(Z_var_add) * torch.randn(
                Z_next_mean.shape, generator=gen, device=device
            )
            if sample_Z else Z_next_mean
        )
        Z = survived * Z_new + (1.0 - survived) * Z
        logit = model.outcome_logit(Z, covariates[:, t, :])
        p_Y = torch.sigmoid(logit)
        new_event = survived * p_Y
        cumulative = cumulative + new_event
        survived = survived * (1.0 - p_Y)
        Z_path.append(Z)
        prob_path.append(p_Y)
        cum_path.append(cumulative.clone())

    return {
        "Z": torch.stack(Z_path, dim=1),
        "prob_Y": torch.stack(prob_path, dim=1),
        "cumulative_risk": torch.stack(cum_path, dim=1),
        "marginal_risk_T": cumulative.mean(),
    }


def _patient_cluster_bootstrap_idx(
    subject_ids: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Cluster bootstrap on subject_id: resample patients then take all stays."""
    unique_patients, inverse = np.unique(subject_ids, return_inverse=True)
    sampled = rng.integers(0, len(unique_patients), size=len(unique_patients))
    return np.concatenate([np.where(inverse == p)[0] for p in sampled])


def dose_response_curve(
    model: BaseLatentSSM,
    bootstrap_drivers: Tensor,
    bootstrap_covariates: Tensor,
    n_bins: int,
    bin_slice: slice,
    target_bins: Sequence[int],
    subject_ids: np.ndarray,
    C_static: Optional[Tensor] = None,
    n_bootstrap: int = 100,
    sample_Z: bool = True,
    seed: int = 0,
) -> dict:
    """Sweep intervene_bin to build a dose-response curve with cluster bootstrap CIs."""
    rng = np.random.default_rng(seed)
    risks_per_bin: list[list[float]] = []
    for k in target_bins:
        boot = []
        for _ in range(n_bootstrap):
            idx_np = _patient_cluster_bootstrap_idx(subject_ids, rng)
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=bootstrap_drivers.device)
            res = simulate_g_formula(
                model=model,
                bootstrap_drivers=bootstrap_drivers[idx],
                bootstrap_covariates=bootstrap_covariates[idx],
                intervene_bin=k,
                n_bins=n_bins,
                bin_slice=bin_slice,
                C_static=C_static[idx] if C_static is not None else None,
                sample_Z=sample_Z,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            boot.append(float(res["marginal_risk_T"].cpu()))
        risks_per_bin.append(boot)
    risk_mat = torch.tensor(risks_per_bin)
    return {
        "bins": list(target_bins),
        "risk_mean": risk_mat.mean(dim=1),
        "risk_ci_low": risk_mat.quantile(0.025, dim=1),
        "risk_ci_high": risk_mat.quantile(0.975, dim=1),
        "risk_raw": risk_mat,
    }
