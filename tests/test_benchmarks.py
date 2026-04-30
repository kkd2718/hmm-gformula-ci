"""Smoke tests for the 3 benchmark methods (interface contract + small fit)."""
import numpy as np
import torch

from src.benchmarks import (
    StandardGFormula, XuGLMM, VEMSSMBenchmark, DoseResponseResult,
)
from src.benchmarks.base import BenchmarkMethod
from src.data.ards import ARDSCohort
from src.utils.seeds import set_seed


def _toy_cohort(N=20, T=6, K=3, p_dyn=2, p_stat=1, seed=0) -> ARDSCohort:
    """Synthesize a tiny ARDSCohort with all required fields populated."""
    set_seed(seed)
    Y = (torch.rand(N, T, 1) < 0.15).float()
    A_bin = torch.zeros(N, T, K)
    bins = torch.randint(0, K, (N, T))
    A_bin.scatter_(2, bins.unsqueeze(-1), 1.0)
    L_dyn = torch.randn(N, T, p_dyn) * 0.5
    C_static = torch.randn(N, p_stat) * 0.5
    at_risk = torch.ones(N, T, 1)
    t_norm = (torch.arange(T).float() / max(T - 1, 1))[None, :, None].expand(N, T, 1).contiguous()
    C_broadcast = C_static.unsqueeze(1).expand(-1, T, -1)
    drivers = torch.cat([A_bin, L_dyn, C_broadcast], dim=-1)
    covariates = torch.cat([drivers, t_norm], dim=-1)
    edges = np.linspace(1.0, 30.0, K + 1)
    subject_ids = np.arange(N)
    return ARDSCohort(
        Y=Y, A_bin=A_bin, L_dyn=L_dyn, C_static=C_static,
        at_risk=at_risk, t_norm=t_norm, drivers=drivers, covariates=covariates,
        bin_edges_mp=edges, stay_ids=np.arange(N), subject_ids=subject_ids,
        severity_label=np.array(["moderate"] * N),
        feature_layout={
            "n_bins": K, "n_dyn": p_dyn, "n_static": p_stat,
            "drivers_width": K + p_dyn + p_stat,
            "covariates_width": K + p_dyn + p_stat + 1,
            "tv_cols": [f"L{i}" for i in range(p_dyn)],
            "static_cols": [f"C{i}" for i in range(p_stat)],
            "mp_bin_edges": edges.tolist(),
            "log_mp_mu": 0.0, "log_mp_sd": 1.0,
            "bin_slice": (0, K),
        },
    )


def _check_dose_response(method: BenchmarkMethod, cohort: ARDSCohort) -> None:
    method.fit(cohort)
    res = method.dose_response(cohort, target_bins=[0, 1, 2], n_bootstrap=3, seed=0)
    assert isinstance(res, DoseResponseResult)
    assert len(res.risk_mean) == 3
    assert (res.risk_mean >= 0).all() and (res.risk_mean <= 1).all()
    assert (res.risk_ci_low <= res.risk_mean + 1e-6).all()
    assert (res.risk_ci_high >= res.risk_mean - 1e-6).all()


def test_standard_gformula():
    _check_dose_response(StandardGFormula(), _toy_cohort())


def test_xu_glmm():
    _check_dose_response(XuGLMM(), _toy_cohort())


def test_vem_ssm_smoke():
    from src.training.variational_em import TrainingConfig
    from src.benchmarks.proposed import VEMConfig
    cfg = VEMConfig(training=TrainingConfig(
        n_epochs=5, learning_rate=5e-2, n_mc_samples=1,
        smoothness_lambda=0.0, grad_clip=5.0, print_every=100,
        early_stop_patience=100, early_stop_tol=0.0,
    ))
    _check_dose_response(VEMSSMBenchmark(cfg), _toy_cohort())
