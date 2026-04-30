"""Regression tests: counterfactual A propagates through Z (option B) and L."""
import torch
import numpy as np

from src.models.linear_gaussian_ssm import LinearGaussianSSM, SSMConfig
from src.inference.g_formula import simulate_g_formula
from src.utils.seeds import set_seed


def _make_model(option_b: bool, K=5, p_dyn=3, p_stat=2, seed=0):
    set_seed(seed)
    cfg = SSMConfig(
        n_bins=K, n_dyn_covariates=p_dyn, n_static_covariates=p_stat,
        z_depends_on_treatment_lag=option_b, fit_time_effect=False,
    )
    model = LinearGaussianSSM(cfg)
    # Hand-set non-zero coefficients so trajectories visibly differ across bins
    with torch.no_grad():
        model.gamma_A.data = torch.linspace(-1.0, 1.0, K)
        if model.delta_A is not None:
            model.delta_A.weight.data = torch.linspace(-0.8, 0.8, K).unsqueeze(0).expand(p_dyn, K).clone()
        model.beta_A.data = torch.linspace(-1.0, 1.0, K)
    return model


def _baseline_inputs(N=30, T=8, K=5, p_dyn=3, p_stat=2, seed=1):
    set_seed(seed)
    A = torch.zeros(N, T, K)
    A.scatter_(2, torch.randint(0, K, (N, T, 1)), 1.0)
    L0 = torch.randn(N, p_dyn) * 0.5
    V = torch.randn(N, p_stat) * 0.5
    t_norm = (torch.arange(T).float() / max(T - 1, 1))[None, :, None].expand(N, T, 1).contiguous()
    return A, L0, V, t_norm


def test_L_trajectory_differs_across_bins():
    """L_t must differ across counterfactual A bins (delta_A path)."""
    K, p_dyn, p_stat = 5, 3, 2
    model = _make_model(option_b=True, K=K, p_dyn=p_dyn, p_stat=p_stat)
    A, L0, V, t_norm = _baseline_inputs(K=K, p_dyn=p_dyn, p_stat=p_stat)

    res_low = simulate_g_formula(
        model, A, L0, V, t_norm, intervene_bin=0,
        n_bins=K, sample_Z=False, sample_L=False, seed=42,
    )
    res_high = simulate_g_formula(
        model, A, L0, V, t_norm, intervene_bin=K - 1,
        n_bins=K, sample_Z=False, sample_L=False, seed=42,
    )
    L_low = res_low["L"].cpu().numpy()
    L_high = res_high["L"].cpu().numpy()
    # Aggregate L difference at t = T-1 should be large
    diff = float(np.abs(L_low[:, -1, :] - L_high[:, -1, :]).mean())
    assert diff > 0.05, f"L trajectories nearly identical across bins (diff={diff})"


def test_Z_responds_to_A_under_option_B_only():
    K, p_dyn, p_stat = 5, 3, 2
    A, L0, V, t_norm = _baseline_inputs(K=K, p_dyn=p_dyn, p_stat=p_stat)
    diffs = {}
    for label, option_b in [("A", False), ("B", True)]:
        model = _make_model(option_b=option_b, K=K, p_dyn=p_dyn, p_stat=p_stat)
        # Zero out all non-A paths so only γ_A·A_lag remains for Z (option B)
        with torch.no_grad():
            if model.gamma_L is not None:
                model.gamma_L.weight.data.zero_()
            if model.gamma_V_dyn is not None:
                model.gamma_V_dyn.weight.data.zero_()
            if model.delta_A is not None:
                model.delta_A.weight.data.zero_()
            model.delta_Z.data.zero_()
        res_low = simulate_g_formula(
            model, A, L0, V, t_norm, intervene_bin=0,
            n_bins=K, sample_Z=False, sample_L=False, seed=42,
        )
        res_high = simulate_g_formula(
            model, A, L0, V, t_norm, intervene_bin=K - 1,
            n_bins=K, sample_Z=False, sample_L=False, seed=42,
        )
        diffs[label] = float(
            np.abs(res_low["Z"][:, -1, :].cpu().numpy() - res_high["Z"][:, -1, :].cpu().numpy()).mean()
        )
    assert diffs["B"] > 1e-4, f"Z trajectory should respond to A under option B (diff={diffs['B']})"
    assert diffs["A"] < 1e-6, f"Z trajectory must be invariant to A under option A (diff={diffs['A']})"
    assert diffs["B"] > diffs["A"] * 100, "Option B effect on Z should dominate option A"


def test_marginal_risk_bounded():
    """Cumulative incidence must stay in [0, 1]."""
    K, p_dyn, p_stat = 5, 3, 2
    model = _make_model(option_b=True, K=K, p_dyn=p_dyn, p_stat=p_stat)
    A, L0, V, t_norm = _baseline_inputs(K=K, p_dyn=p_dyn, p_stat=p_stat)
    res = simulate_g_formula(
        model, A, L0, V, t_norm, intervene_bin=2,
        n_bins=K, sample_Z=True, sample_L=True, seed=0,
    )
    risk = float(res["marginal_risk_T"].cpu())
    assert 0.0 <= risk <= 1.0, f"marginal_risk_T outside [0,1]: {risk}"
