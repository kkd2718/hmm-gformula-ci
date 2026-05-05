"""VEM-SSM g-formula benchmark wrapper (proposed; time-varying latent state).

Two model variants controlled by SSMConfig.z_depends_on_treatment_lag:
    Option A (False) — Z_t exogenous to treatment (Xu's b_i extended to AR latent)
    Option B (True)  — Z_t includes A_{t-1} dependency (full SSM, primary spec)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..data.ards import ARDSCohort
from ..models.linear_gaussian_ssm import LinearGaussianSSM, SSMConfig
from ..models.variational_posterior import StructuredGaussianMarkovPosterior, QConfig
from ..training.variational_em import train_vem, TrainingConfig
from ..inference.g_formula import simulate_g_formula
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from ._resample import cluster_bootstrap_indices, slice_cohort


@dataclass
class VEMConfig:
    """Hyperparameters bundling SSM and training configs.

    z_depends_on_treatment_lag: True = option B (full); False = option A.
    """
    training: TrainingConfig = field(default_factory=TrainingConfig)
    z_depends_on_treatment_lag: bool = True


class VEMSSMBenchmark(BenchmarkMethod):
    """Variational EM linear-Gaussian SSM with parametric NICE g-formula."""

    method_name = "vem_ssm"

    def __init__(self, config: VEMConfig | None = None) -> None:
        self.config = config or VEMConfig()
        self._model: LinearGaussianSSM | None = None
        self._posterior: StructuredGaussianMarkovPosterior | None = None
        self.last_history = None

    def _new_model(self, cohort: ARDSCohort) -> None:
        layout = cohort.feature_layout
        ssm_cfg = SSMConfig(
            n_bins=layout["n_bins"],
            n_dyn_covariates=layout["n_dyn"],
            n_static_covariates=layout["n_static"],
            z_depends_on_treatment_lag=self.config.z_depends_on_treatment_lag,
            fit_time_effect=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = LinearGaussianSSM(ssm_cfg).to(device)
        self._posterior = StructuredGaussianMarkovPosterior(
            QConfig(
                n_bins=layout["n_bins"],
                n_dyn_covariates=layout["n_dyn"],
                n_static_covariates=layout["n_static"],
            )
        ).to(device)

    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        self._new_model(cohort)
        self.last_history = train_vem(
            model=self._model, posterior=self._posterior,
            Y=cohort.Y, A=cohort.A_bin, L=cohort.L_dyn, V=cohort.C_static,
            at_risk=cohort.at_risk, t_norm=cohort.t_norm,
            config=self.config.training, verbose=False,
        )

    def fitted_param_snapshot(self) -> dict:
        """Return numpy snapshot of identifiable bin-indexed and SSM parameters
        for post-hoc diagnostics (collapsed / explosive coefficient detection)."""
        if self._model is None:
            return {}
        m = self._model
        snap = {
            "psi": float(m.psi.detach().cpu()),
            "sigma_Z": float(m.sigma_Z.detach().cpu()),
            "beta_0": float(m.beta_0_param.detach().cpu()),
            "beta_Z": float(m.beta_Z.detach().cpu()),
            "beta_A": m.beta_A.detach().cpu().numpy().tolist(),
            "gamma_A": m.gamma_A.detach().cpu().numpy().tolist(),
        }
        if m.delta_Z is not None:
            snap["delta_Z"] = m.delta_Z.detach().cpu().numpy().tolist()
        if m.alpha_L is not None:
            snap["alpha_L"] = m.alpha_L.detach().cpu().numpy().tolist()
        if m.delta_A is not None:
            snap["delta_A"] = m.delta_A.weight.detach().cpu().numpy().tolist()
        if m.sigma_L is not None:
            snap["sigma_L"] = m.sigma_L.detach().cpu().numpy().tolist()
        return snap

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 100, seed: int = 0, refit: bool = False,
    ) -> DoseResponseResult:
        """Outer bootstrap × inner bin loop. refit=True retrains per replicate."""
        rng = np.random.default_rng(seed)
        if not refit and self._model is None:
            self.fit(cohort)
        K_bins = cohort.feature_layout["n_bins"]
        K_targets = len(target_bins)
        risk_mat = np.zeros((K_targets, n_bootstrap), dtype=np.float64)
        for b in range(n_bootstrap):
            idx = cluster_bootstrap_indices(cohort.subject_ids, rng)
            boot_cohort = slice_cohort(cohort, idx)
            if refit:
                self.fit(boot_cohort)
            device = next(self._model.parameters()).device
            A_dev = boot_cohort.A_bin.to(device)
            V_dev = boot_cohort.C_static.to(device)
            t_dev = boot_cohort.t_norm.to(device)
            L0_dev = boot_cohort.L_dyn[:, 0, :].to(device)
            for ki, k in enumerate(target_bins):
                res = simulate_g_formula(
                    model=self._model,
                    A_baseline=A_dev,
                    L_baseline_t0=L0_dev,
                    V=V_dev,
                    t_norm=t_dev,
                    intervene_bin=k,
                    n_bins=K_bins,
                    sample_Z=True,
                    sample_L=True,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                risk_mat[ki, b] = float(res["marginal_risk_T"].cpu())
        return DoseResponseResult(
            bins=list(target_bins),
            bin_centers_J_min=bin_centers_J_min(cohort),
            risk_mean=risk_mat.mean(axis=1),
            risk_ci_low=np.quantile(risk_mat, 0.025, axis=1),
            risk_ci_high=np.quantile(risk_mat, 0.975, axis=1),
            risk_raw=risk_mat,
            method_name=self.method_name,
        )
