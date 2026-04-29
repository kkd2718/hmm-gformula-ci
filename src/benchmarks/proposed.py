"""VEM-SSM g-formula benchmark wrapper (proposed; time-varying latent state)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..data.ards import ARDSCohort
from ..models.linear_gaussian_ssm import LinearGaussianSSM, SSMConfig
from ..models.variational_posterior import StructuredGaussianMarkovPosterior, QConfig
from ..training.variational_em import train_vem, TrainingConfig
from ..inference.g_formula import simulate_g_formula
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from ._resample import cluster_bootstrap_indices, slice_cohort


@dataclass
class VEMConfig:
    """Hyperparameters bundling SSM, posterior, and training configs."""
    training: TrainingConfig = field(default_factory=TrainingConfig)


class VEMSSMBenchmark(BenchmarkMethod):
    """Variational EM linear-Gaussian SSM with parametric g-formula (proposed).

    Refit-bootstrap is supported but expensive (each fit is O(epochs * N*T)),
    so default n_bootstrap is small. For a published primary CI use refit=False
    with cluster bootstrap of the simulation step only; theta uncertainty is
    reported separately on a refit-bootstrap subset (Keil et al. 2014).

    `last_history` retains the TrainingHistory of the most recent fit() call
    for ELBO-trajectory diagnostics.
    """

    method_name = "vem_ssm"

    def __init__(self, config: VEMConfig | None = None) -> None:
        self.config = config or VEMConfig()
        self._model: LinearGaussianSSM | None = None
        self._posterior: StructuredGaussianMarkovPosterior | None = None
        self.last_history = None

    def _new_model(self, cohort: ARDSCohort) -> None:
        L = cohort.feature_layout
        ssm_cfg = SSMConfig(
            n_bins=L["n_bins"],
            n_dyn_covariates=L["n_dyn"],
            n_static_covariates=L["n_static"],
            fit_time_effect=True,
        )
        self._model = LinearGaussianSSM(ssm_cfg)
        p = ssm_cfg.n_bins + ssm_cfg.n_dyn_covariates + ssm_cfg.n_static_covariates
        self._posterior = StructuredGaussianMarkovPosterior(
            QConfig(n_drivers=p, n_covariates=p + 1)
        )

    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        self._new_model(cohort)
        self.last_history = train_vem(
            model=self._model, posterior=self._posterior,
            Y=cohort.Y, drivers=cohort.drivers, covariates=cohort.covariates,
            at_risk=cohort.at_risk, C_static=cohort.C_static,
            config=self.config.training, verbose=False,
        )

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 100, seed: int = 0, refit: bool = False,
    ) -> DoseResponseResult:
        """Cluster-bootstrap dose-response. refit=True is supported but costs
        one full ELBO training run per replicate; use sparingly."""
        rng = np.random.default_rng(seed)
        if not refit and self._model is None:
            self.fit(cohort)
        L = cohort.feature_layout
        bin_lo, bin_hi = L["bin_slice"]
        risks_per_bin: list[list[float]] = []
        for k in target_bins:
            boot = []
            for _ in range(n_bootstrap):
                idx = cluster_bootstrap_indices(cohort.subject_ids, rng)
                boot_cohort = slice_cohort(cohort, idx)
                if refit:
                    self.fit(boot_cohort)
                res = simulate_g_formula(
                    model=self._model,
                    bootstrap_drivers=boot_cohort.drivers,
                    bootstrap_covariates=boot_cohort.covariates,
                    intervene_bin=k,
                    n_bins=L["n_bins"],
                    bin_slice=slice(bin_lo, bin_hi),
                    C_static=boot_cohort.C_static,
                    sample_Z=True,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                boot.append(float(res["marginal_risk_T"].cpu()))
            risks_per_bin.append(boot)
        risk_mat = np.asarray(risks_per_bin)
        return DoseResponseResult(
            bins=list(target_bins),
            bin_centers_J_min=bin_centers_J_min(cohort),
            risk_mean=risk_mat.mean(axis=1),
            risk_ci_low=np.quantile(risk_mat, 0.025, axis=1),
            risk_ci_high=np.quantile(risk_mat, 0.975, axis=1),
            risk_raw=risk_mat,
            method_name=self.method_name,
        )
