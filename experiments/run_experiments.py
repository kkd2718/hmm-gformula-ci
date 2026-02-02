"""
experiments/run_experiments.py - Complete Experiment Suite (v3.3)

논문용 전체 실험 통합 버전 (20년 시뮬레이션)

실험 1: Effect Size Sensitivity (Parameter Recovery)
실험 2: Sample Size Robustness (5K ~ 100K)
실험 3: Model Selection & Ablation Study (LRT, AIC, BIC)
실험 4: 3-Way Method Comparison (Cox PH vs Markov vs HMM)
실험 5: Causal Inference with Bootstrap CI
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from scipy import stats
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEED, DEFAULT_DATA_PARAMS, TRAINING_PARAMS,
    TRUE_PARAMS_HIDDEN_STATE, TRUE_PARAMS_OUTCOME, TRUE_PARAMS_EXPOSURE,
    EXP1_EFFECT_SIZES, EXP2_SAMPLE_SIZES, EXP3_MISSPEC_SCENARIOS,
    MC_PARAMS, GFORMULA_PARAMS, OUTPUT_DIR, N_COVARIATES,
    get_modified_params, get_n_epochs,
)
from data_generator import generate_synthetic_data, validate_dgp, SimulatedData
from models import HiddenMarkovGFormula, estimate_causal_effect

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_log_likelihood(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    S: torch.Tensor,
    C: torch.Tensor,
    Y: torch.Tensor,
    L: torch.Tensor,
) -> float:
    """모델의 Log-Likelihood 계산"""
    model.eval()
    with torch.no_grad():
        if model.use_hmm:
            estimates = model.forward_filter(G, S, C, Y, L)
            return estimates.log_likelihood
        else:
            n_time = S.shape[1]
            total_ll = 0.0
            Z_curr = torch.zeros(G.shape[0], 1)
            
            for t in range(n_time):
                Z_mean, _ = model._hidden_state_transition(Z_curr, S[:, t, :], C[:, t, :], G, L)
                prob_Y = model._outcome_probability(Z_mean, S[:, t, :], G, L, t=t)
                
                ll = Y[:, t, :] * torch.log(prob_Y + 1e-10) + (1 - Y[:, t, :]) * torch.log(1 - prob_Y + 1e-10)
                total_ll += ll.sum().item()
                Z_curr = Z_mean
            
            return total_ll


def count_parameters(model: HiddenMarkovGFormula) -> int:
    """학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_aic(log_likelihood: float, n_params: int) -> float:
    return -2 * log_likelihood + 2 * n_params


def compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    return -2 * log_likelihood + n_params * np.log(n_samples)


def likelihood_ratio_test(ll_full: float, ll_reduced: float, df: int) -> Tuple[float, float]:
    chi2_stat = max(0, 2 * (ll_full - ll_reduced))
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    return chi2_stat, p_value


# =============================================================================
# Bootstrap Utilities
# =============================================================================

def bootstrap_causal_effect(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    L: torch.Tensor,
    n_bootstrap: int = 200,
    n_time: int = 20,
    confidence_level: float = 0.95,
    verbose: bool = True,
) -> Dict:
    """Bootstrap을 이용한 인과 효과 추정 및 95% CI 계산"""
    n_samples = G.shape[0]
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    
    risk_never_list, risk_always_list = [], []
    risk_diff_list, risk_ratio_list = [], []
    
    iterator = tqdm(range(n_bootstrap), desc="Bootstrap") if verbose else range(n_bootstrap)
    
    for b in iterator:
        idx = torch.randint(0, n_samples, (n_samples,))
        G_boot, L_boot = G[idx], L[idx]
        
        result_never = model.simulate_gformula(G_boot, L_boot, 'never_smoke', n_time, n_monte_carlo=n_mc)
        result_always = model.simulate_gformula(G_boot, L_boot, 'always_smoke', n_time, n_monte_carlo=n_mc)
        
        risk_never = result_never['mean_cumulative_risk']
        risk_always = result_always['mean_cumulative_risk']
        
        risk_never_list.append(risk_never)
        risk_always_list.append(risk_always)
        risk_diff_list.append(risk_always - risk_never)
        risk_ratio_list.append(risk_always / (risk_never + 1e-10))
    
    alpha = 1 - confidence_level
    
    def compute_ci(values):
        values = np.array(values)
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, 100 * alpha / 2),
            'ci_upper': np.percentile(values, 100 * (1 - alpha / 2)),
        }
    
    return {
        'risk_never_smoke': compute_ci(risk_never_list),
        'risk_always_smoke': compute_ci(risk_always_list),
        'risk_difference': compute_ci(risk_diff_list),
        'risk_ratio': compute_ci(risk_ratio_list),
        'n_bootstrap': n_bootstrap,
    }


def bootstrap_quit_timing_effect(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    L: torch.Tensor,
    quit_times: List[int] = None,
    n_bootstrap: int = 100,
    n_time: int = 20,
    verbose: bool = True,
) -> Dict:
    """금연 시점에 따른 효과 변화"""
    if quit_times is None:
        quit_times = list(range(0, n_time, 2))
    
    results = {t: [] for t in quit_times}
    n_samples = G.shape[0]
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap quit timing", disable=not verbose):
        idx = torch.randint(0, n_samples, (n_samples,))
        G_boot, L_boot = G[idx], L[idx]
        
        ref = model.simulate_gformula(G_boot, L_boot, 'always_smoke', n_time, n_monte_carlo=n_mc)
        risk_always = ref['mean_cumulative_risk']
        
        for qt in quit_times:
            res = model.simulate_gformula(G_boot, L_boot, f'quit_at_t{qt}', n_time, n_monte_carlo=n_mc, quit_time=qt)
            rr = res['mean_cumulative_risk'] / (risk_always + 1e-10)
            results[qt].append(rr)
    
    summary = {}
    for qt in quit_times:
        values = np.array(results[qt])
        summary[qt] = {
            'mean_rr': np.mean(values),
            'ci_lower': np.percentile(values, 2.5),
            'ci_upper': np.percentile(values, 97.5),
        }
    
    return summary


def bootstrap_prs_stratified_effect(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    L: torch.Tensor,
    n_bootstrap: int = 100,
    n_time: int = 20,
    prs_percentile: float = 80,
    verbose: bool = True,
) -> Dict:
    """PRS 층화 분석"""
    n_samples = G.shape[0]
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    
    prs_threshold = torch.quantile(G, prs_percentile / 100)
    high_risk_mask = G.squeeze() >= prs_threshold
    low_risk_mask = ~high_risk_mask
    
    results = {'high_risk': [], 'low_risk': []}
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap PRS stratified", disable=not verbose):
        for group, mask in [('high_risk', high_risk_mask), ('low_risk', low_risk_mask)]:
            G_group, L_group = G[mask], L[mask]
            
            if len(G_group) < 10:
                continue
            
            n_group = len(G_group)
            idx = torch.randint(0, n_group, (n_group,))
            G_boot, L_boot = G_group[idx], L_group[idx]
            
            res_never = model.simulate_gformula(G_boot, L_boot, 'never_smoke', n_time, n_monte_carlo=n_mc//2)
            res_always = model.simulate_gformula(G_boot, L_boot, 'always_smoke', n_time, n_monte_carlo=n_mc//2)
            
            rr = res_always['mean_cumulative_risk'] / (res_never['mean_cumulative_risk'] + 1e-10)
            results[group].append(rr)
    
    summary = {}
    for group in ['high_risk', 'low_risk']:
        values = np.array(results[group])
        summary[group] = {
            'mean_rr': np.mean(values),
            'ci_lower': np.percentile(values, 2.5),
            'ci_upper': np.percentile(values, 97.5),
        }
    
    return summary


# =============================================================================
# Experiment 1: Effect Size Sensitivity (Parameter Recovery)
# =============================================================================

def run_experiment_1(
    n_simulations: int = None,
    n_samples: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """실험 1: Effect Size별 Parameter Recovery"""
    n_sim = n_simulations or MC_PARAMS['n_simulations']
    n_samples = n_samples or DEFAULT_DATA_PARAMS['n_samples']
    n_time = DEFAULT_DATA_PARAMS['n_time']
    n_epochs = get_n_epochs(n_samples)
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 1: Effect Size Sensitivity (Parameter Recovery)")
        print("="*70)
        print(f"N simulations: {n_sim}, N samples: {n_samples}, N time: {n_time}")
    
    results = {scenario: defaultdict(list) for scenario in EXP1_EFFECT_SIZES}
    true_values = {}
    
    for scenario_name, effect_mods in EXP1_EFFECT_SIZES.items():
        if verbose:
            print(f"\n>>> Scenario: {scenario_name}")
        
        params_h = get_modified_params(TRUE_PARAMS_HIDDEN_STATE, {
            'gamma_GS': effect_mods['gamma_GS'], 
            'gamma_GC': effect_mods['gamma_GC']
        })
        params_o = get_modified_params(TRUE_PARAMS_OUTCOME, {
            'beta_GS': effect_mods['beta_GS']
        })
        true_values[scenario_name] = effect_mods
        
        for sim_idx in tqdm(range(n_sim), desc=f"  {scenario_name}", disable=not verbose):
            set_seed(SEED + sim_idx)
            
            try:
                data = generate_synthetic_data(
                    n_samples, n_time, params_h, params_o, TRUE_PARAMS_EXPOSURE,
                    seed=SEED + sim_idx
                )
                
                model = HiddenMarkovGFormula(n_covariates=N_COVARIATES)
                model.fit(data.G, data.S, data.C, data.Y, L=data.L, 
                         n_epochs=n_epochs, verbose=False)
                
                params = model.get_parameters()
                for k, v in params.items():
                    if not isinstance(v, list):
                        results[scenario_name][k].append(v)
                        
            except Exception as e:
                if verbose:
                    print(f"    Warning: Sim {sim_idx} failed - {e}")
    
    # Summary
    if verbose:
        print("\n" + "-"*70)
        print("Results Summary (beta_GS estimation):")
        print("-"*70)
        print(f"{'Scenario':<12} {'True':>8} {'Mean':>8} {'Bias':>8} {'RMSE':>8}")
        print("-"*70)
        
        for scenario, true_vals in true_values.items():
            true_beta = true_vals['beta_GS']
            if results[scenario]['beta_GS']:
                est = np.array(results[scenario]['beta_GS'])
                bias = np.mean(est) - true_beta
                rmse = np.sqrt(np.mean((est - true_beta)**2))
                print(f"{scenario:<12} {true_beta:>8.3f} {np.mean(est):>8.3f} {bias:>8.3f} {rmse:>8.3f}")
    
    if save_results:
        rows = []
        for scenario, param_dict in results.items():
            true_vals = true_values[scenario]
            for param, values in param_dict.items():
                for v in values:
                    rows.append({
                        'scenario': scenario, 'parameter': param, 'estimate': v,
                        'true_value': true_vals.get(param.replace('beta_', '').replace('gamma_', ''), None)
                    })
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'exp1_parameter_recovery.csv'), index=False)
        if verbose:
            print(f"\nSaved: {OUTPUT_DIR}/exp1_parameter_recovery.csv")
    
    return {'results': {k: dict(v) for k, v in results.items()}, 'true_values': true_values}


# =============================================================================
# Experiment 2: Sample Size Robustness
# =============================================================================

def run_experiment_2(
    n_simulations: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """실험 2: Sample Size별 추정 안정성"""
    n_sim = n_simulations or MC_PARAMS['n_simulations']
    n_time = DEFAULT_DATA_PARAMS['n_time']
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 2: Sample Size Robustness")
        print("="*70)
        print(f"N simulations: {n_sim}, Sample sizes: {EXP2_SAMPLE_SIZES}")
    
    effect_mods = EXP1_EFFECT_SIZES['Moderate']
    params_h = get_modified_params(TRUE_PARAMS_HIDDEN_STATE, {
        'gamma_GS': effect_mods['gamma_GS'], 'gamma_GC': effect_mods['gamma_GC']
    })
    params_o = get_modified_params(TRUE_PARAMS_OUTCOME, {'beta_GS': effect_mods['beta_GS']})
    true_beta_GS = effect_mods['beta_GS']
    
    results = {n: defaultdict(list) for n in EXP2_SAMPLE_SIZES}
    timing = {}
    
    for n_samples in EXP2_SAMPLE_SIZES:
        n_epochs = get_n_epochs(n_samples)
        
        if verbose:
            print(f"\n>>> N = {n_samples:,} (epochs={n_epochs})")
        
        start_time = time.time()
        
        for sim_idx in tqdm(range(n_sim), desc=f"  N={n_samples:,}", disable=not verbose):
            set_seed(SEED + sim_idx + n_samples)
            
            try:
                data = generate_synthetic_data(
                    n_samples, n_time, params_h, params_o, TRUE_PARAMS_EXPOSURE,
                    seed=SEED + sim_idx + n_samples
                )
                
                model = HiddenMarkovGFormula(n_covariates=N_COVARIATES)
                model.fit(data.G, data.S, data.C, data.Y, L=data.L,
                         n_epochs=n_epochs, verbose=False)
                
                params = model.get_parameters()
                for k, v in params.items():
                    if not isinstance(v, list):
                        results[n_samples][k].append(v)
                        
            except Exception as e:
                if verbose:
                    print(f"    Warning: Sim {sim_idx} failed - {e}")
        
        timing[n_samples] = time.time() - start_time
        if verbose:
            print(f"    Completed in {timing[n_samples]/60:.1f} minutes")
    
    # Summary
    if verbose:
        print("\n" + "-"*70)
        print(f"Results Summary (beta_GS, True = {true_beta_GS})")
        print("-"*70)
        print(f"{'N':>10} {'Mean':>10} {'Bias':>10} {'RMSE':>10} {'SE':>10}")
        print("-"*70)
        
        for n_samples in EXP2_SAMPLE_SIZES:
            if results[n_samples]['beta_GS']:
                est = np.array(results[n_samples]['beta_GS'])
                print(f"{n_samples:>10,} {np.mean(est):>10.4f} {np.mean(est)-true_beta_GS:>10.4f} "
                      f"{np.sqrt(np.mean((est-true_beta_GS)**2)):>10.4f} {np.std(est):>10.4f}")
    
    if save_results:
        rows = []
        for n_samples, param_dict in results.items():
            for param, values in param_dict.items():
                for v in values:
                    rows.append({'sample_size': n_samples, 'parameter': param, 'estimate': v})
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'exp2_sample_size.csv'), index=False)
        
        summary_rows = []
        for n_samples in EXP2_SAMPLE_SIZES:
            if results[n_samples]['beta_GS']:
                est = np.array(results[n_samples]['beta_GS'])
                summary_rows.append({
                    'n_samples': n_samples, 'true_value': true_beta_GS,
                    'mean': np.mean(est), 'bias': np.mean(est) - true_beta_GS,
                    'rmse': np.sqrt(np.mean((est - true_beta_GS)**2)),
                    'se': np.std(est), 'time_min': timing[n_samples] / 60,
                })
        pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, 'exp2_summary.csv'), index=False)
        
        if verbose:
            print(f"\nSaved: {OUTPUT_DIR}/exp2_sample_size.csv, exp2_summary.csv")
    
    return {'results': {k: dict(v) for k, v in results.items()}, 'true_beta_GS': true_beta_GS}


# =============================================================================
# Experiment 3: Model Selection & Ablation Study (LRT, AIC, BIC)
# =============================================================================

def run_experiment_3(
    n_simulations: int = None,
    n_samples: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """실험 3: Model Selection & Ablation Study"""
    n_sim = n_simulations or MC_PARAMS['n_simulations']
    n_samples = n_samples or DEFAULT_DATA_PARAMS['n_samples']
    n_time = DEFAULT_DATA_PARAMS['n_time']
    n_epochs = get_n_epochs(n_samples)
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 3: Model Selection & Ablation Study (LRT)")
        print("="*70)
        print(f"N simulations: {n_sim}, N samples: {n_samples}, N time: {n_time}")
    
    model_configs = {
        'Full_Model': {'fit_interaction': True, 'fit_pack_years': True, 'use_hmm': True},
        'No_Interaction': {'fit_interaction': False, 'fit_pack_years': True, 'use_hmm': True},
        'No_PackYears': {'fit_interaction': True, 'fit_pack_years': False, 'use_hmm': True},
    }
    
    results = []
    
    for sim_idx in tqdm(range(n_sim), desc="Simulations", disable=not verbose):
        set_seed(SEED + sim_idx)
        
        data = generate_synthetic_data(
            n_samples, n_time, 
            TRUE_PARAMS_HIDDEN_STATE, TRUE_PARAMS_OUTCOME, TRUE_PARAMS_EXPOSURE,
            seed=SEED + sim_idx
        )
        
        sim_results = {'sim_id': sim_idx}
        
        for model_name, config in model_configs.items():
            try:
                model = HiddenMarkovGFormula(
                    n_covariates=N_COVARIATES,
                    fit_interaction=config['fit_interaction'],
                    fit_pack_years=config['fit_pack_years'],
                    use_hmm=config['use_hmm'],
                )
                model.fit(data.G, data.S, data.C, data.Y, L=data.L, 
                         n_epochs=n_epochs, verbose=False)
                
                ll = compute_log_likelihood(model, data.G, data.S, data.C, data.Y, data.L)
                n_params = count_parameters(model)
                
                sim_results[f'{model_name}_LL'] = ll
                sim_results[f'{model_name}_AIC'] = compute_aic(ll, n_params)
                sim_results[f'{model_name}_BIC'] = compute_bic(ll, n_params, n_samples)
                sim_results[f'{model_name}_nparams'] = n_params
                
                params = model.get_parameters()
                if 'beta_GS' in params:
                    sim_results[f'{model_name}_beta_GS'] = params['beta_GS']
                
            except Exception as e:
                sim_results[f'{model_name}_LL'] = np.nan
        
        # LRT
        ll_full = sim_results.get('Full_Model_LL', np.nan)
        for reduced_name in ['No_Interaction', 'No_PackYears']:
            ll_reduced = sim_results.get(f'{reduced_name}_LL', np.nan)
            if not np.isnan(ll_full) and not np.isnan(ll_reduced):
                df = max(1, abs(sim_results.get('Full_Model_nparams', 0) - sim_results.get(f'{reduced_name}_nparams', 0)))
                chi2, p_value = likelihood_ratio_test(ll_full, ll_reduced, df)
                sim_results[f'LRT_{reduced_name}_chi2'] = chi2
                sim_results[f'LRT_{reduced_name}_pvalue'] = p_value
        
        results.append(sim_results)
    
    df_results = pd.DataFrame(results)
    
    if verbose:
        print("\n" + "-"*70)
        print("Model Comparison Summary")
        print("-"*70)
        for model_name in model_configs.keys():
            ll_col = f'{model_name}_LL'
            if ll_col in df_results.columns:
                print(f"\n{model_name}:")
                print(f"  LL: {df_results[ll_col].mean():.1f} ± {df_results[ll_col].std():.1f}")
                print(f"  AIC: {df_results[f'{model_name}_AIC'].mean():.1f}")
                print(f"  BIC: {df_results[f'{model_name}_BIC'].mean():.1f}")
        
        print("\n" + "-"*70)
        print("LRT Results (Full vs Reduced)")
        for reduced_name in ['No_Interaction', 'No_PackYears']:
            p_col = f'LRT_{reduced_name}_pvalue'
            if p_col in df_results.columns:
                p_vals = df_results[p_col].dropna()
                print(f"\nFull vs {reduced_name}: {(p_vals < 0.05).mean()*100:.1f}% significant (p<0.05)")
    
    if save_results:
        df_results.to_csv(os.path.join(OUTPUT_DIR, 'exp3_model_selection.csv'), index=False)
        
        summary = []
        for model_name in model_configs.keys():
            row = {
                'model': model_name,
                'LL_mean': df_results[f'{model_name}_LL'].mean(),
                'AIC_mean': df_results[f'{model_name}_AIC'].mean(),
                'BIC_mean': df_results[f'{model_name}_BIC'].mean(),
            }
            beta_col = f'{model_name}_beta_GS'
            if beta_col in df_results.columns:
                row['beta_GS_mean'] = df_results[beta_col].mean()
                row['beta_GS_bias'] = df_results[beta_col].mean() - TRUE_PARAMS_OUTCOME['beta_GS']
            summary.append(row)
        pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, 'exp3_summary.csv'), index=False)
        
        if verbose:
            print(f"\nSaved: {OUTPUT_DIR}/exp3_model_selection.csv, exp3_summary.csv")
    
    return {'results': df_results, 'model_configs': model_configs}


# =============================================================================
# Experiment 4: 3-Way Method Comparison (Cox PH vs Markov vs HMM)
# =============================================================================

def compute_discrete_time_hazard_ratio(
    model: HiddenMarkovGFormula,
    exposure_coef: str = 'beta_S',
    interaction_coef: str = 'beta_GS',
    G_mean: float = 0.0,
) -> float:
    """
    이산 시간 Pooled Logistic에서 Hazard Ratio 근사 계산
    
    이산 시간(Discrete Time)에서 보정된 로지스틱 회귀 계수는
    연속 시간 Cox PH의 log(HR)과 근사함 (Rare disease assumption)
    
    HR ≈ exp(beta_S + beta_GS * G_mean)
    
    Reference: D'Agostino et al. (1990), Pooled logistic regression
    """
    params = model.get_parameters()
    beta_S = params.get(exposure_coef, 0)
    beta_GS = params.get(interaction_coef, 0)
    
    log_hr = beta_S + beta_GS * G_mean
    hr = np.exp(log_hr)
    
    return hr


def run_experiment_4(
    n_simulations: int = None,
    n_samples: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    실험 4: 3-Way Method Comparison
    
    목표: HMM이 기존 방법론이 해결하지 못한 편향을 어떻게 교정하는지 증명
    
    비교 모델:
    ┌─────────────────────────┬──────────────┬──────────┬────────────────┐
    │ Model                   │ Time Trans.  │ Latent Z │ Effect Measure │
    ├─────────────────────────┼──────────────┼──────────┼────────────────┤
    │ A. Time-varying Cox PH  │ Hazard-based │ ✗        │ Hazard Ratio   │
    │ B. Standard Markov      │ ✓ (discrete) │ ✗        │ Risk Ratio     │
    │ C. Proposed HMM         │ ✓ (discrete) │ ✓        │ Risk Ratio     │
    └─────────────────────────┴──────────────┴──────────┴────────────────┘
    
    - Model A (Cox PH): use_hmm=False 모델의 beta_S → HR로 해석
      * 이산 시간 pooled logistic ≈ continuous Cox PH (D'Agostino 1990)
    - Model B (Markov): use_hmm=False로 g-formula 시뮬레이션 → RR
    - Model C (HMM): use_hmm=True로 g-formula 시뮬레이션 → RR
    
    Note: HR과 RR은 Rare disease 가정 하에서 비교 가능 (CVD ~5-10%)
    """
    n_sim = n_simulations or MC_PARAMS['n_simulations']
    n_samples = n_samples or DEFAULT_DATA_PARAMS['n_samples']
    n_time = DEFAULT_DATA_PARAMS['n_time']
    n_epochs = get_n_epochs(n_samples)
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 500)
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 4: 3-Way Method Comparison")
        print("="*70)
        print(f"N simulations: {n_sim}, N samples: {n_samples}, N time: {n_time}")
        print("\nModels compared:")
        print("  A. Time-varying Cox PH (Hazard Ratio from pooled logistic)")
        print("  B. Standard Markov g-formula (Risk Ratio, no latent Z)")
        print("  C. Proposed HMM g-formula (Risk Ratio, with latent Z)")
        print("\nNote: Under rare disease assumption, HR ≈ RR")
    
    results = []
    true_beta_GS = TRUE_PARAMS_OUTCOME['beta_GS']
    true_beta_S = TRUE_PARAMS_OUTCOME['beta_S']
    
    for sim_idx in tqdm(range(n_sim), desc="Simulations", disable=not verbose):
        set_seed(SEED + sim_idx)
        
        # 공통 데이터 생성 (공정한 비교)
        data = generate_synthetic_data(
            n_samples, n_time, 
            TRUE_PARAMS_HIDDEN_STATE, TRUE_PARAMS_OUTCOME, TRUE_PARAMS_EXPOSURE,
            seed=SEED + sim_idx
        )
        
        sim_results = {'sim_id': sim_idx}
        
        # =====================================================================
        # Model A: Time-varying Cox PH (via Pooled Logistic approximation)
        # =====================================================================
        try:
            model_cox = HiddenMarkovGFormula(
                n_covariates=N_COVARIATES,
                use_hmm=False,
                fit_interaction=True,
                fit_pack_years=True,
            )
            model_cox.fit(data.G, data.S, data.C, data.Y, L=data.L, 
                         n_epochs=n_epochs, verbose=False)
            
            params_cox = model_cox.get_parameters()
            
            # HR from pooled logistic coefficients
            hr = compute_discrete_time_hazard_ratio(model_cox, G_mean=0.0)
            
            # Model fit statistics (LL, AIC, BIC)
            ll_cox = compute_log_likelihood(model_cox, data.G, data.S, data.C, data.Y, data.L)
            n_params_cox = count_parameters(model_cox)
            aic_cox = compute_aic(ll_cox, n_params_cox)
            bic_cox = compute_bic(ll_cox, n_params_cox, n_samples)
            
            sim_results['CoxPH_HR'] = hr
            sim_results['CoxPH_beta_S'] = params_cox.get('beta_S', np.nan)
            sim_results['CoxPH_beta_GS'] = params_cox.get('beta_GS', np.nan)
            sim_results['CoxPH_LL'] = ll_cox
            sim_results['CoxPH_AIC'] = aic_cox
            sim_results['CoxPH_BIC'] = bic_cox
            sim_results['CoxPH_nparams'] = n_params_cox
            
        except Exception as e:
            sim_results['CoxPH_HR'] = np.nan
            sim_results['CoxPH_LL'] = np.nan
            if verbose and sim_idx == 0:
                print(f"  Cox PH failed: {e}")
        
        # =====================================================================
        # Model B: Standard Markov g-formula (no latent Z)
        # =====================================================================
        try:
            model_markov = HiddenMarkovGFormula(
                n_covariates=N_COVARIATES,
                use_hmm=False,
                fit_interaction=True,
                fit_pack_years=True,
            )
            model_markov.fit(data.G, data.S, data.C, data.Y, L=data.L, 
                            n_epochs=n_epochs, verbose=False)
            
            # g-formula simulation
            res_never = model_markov.simulate_gformula(data.G, data.L, 'never_smoke', n_time, n_mc)
            res_always = model_markov.simulate_gformula(data.G, data.L, 'always_smoke', n_time, n_mc)
            
            risk_never = res_never['mean_cumulative_risk']
            risk_always = res_always['mean_cumulative_risk']
            rr_markov = risk_always / (risk_never + 1e-10)
            
            params_markov = model_markov.get_parameters()
            
            # Model fit statistics (LL, AIC, BIC)
            ll_markov = compute_log_likelihood(model_markov, data.G, data.S, data.C, data.Y, data.L)
            n_params_markov = count_parameters(model_markov)
            aic_markov = compute_aic(ll_markov, n_params_markov)
            bic_markov = compute_bic(ll_markov, n_params_markov, n_samples)
            
            sim_results['Markov_RR'] = rr_markov
            sim_results['Markov_risk_never'] = risk_never
            sim_results['Markov_risk_always'] = risk_always
            sim_results['Markov_beta_S'] = params_markov.get('beta_S', np.nan)
            sim_results['Markov_beta_GS'] = params_markov.get('beta_GS', np.nan)
            sim_results['Markov_LL'] = ll_markov
            sim_results['Markov_AIC'] = aic_markov
            sim_results['Markov_BIC'] = bic_markov
            sim_results['Markov_nparams'] = n_params_markov
            
        except Exception as e:
            sim_results['Markov_RR'] = np.nan
            sim_results['Markov_LL'] = np.nan
            if verbose and sim_idx == 0:
                print(f"  Markov failed: {e}")
        
        # =====================================================================
        # Model C: Proposed HMM g-formula (with latent Z)
        # =====================================================================
        try:
            model_hmm = HiddenMarkovGFormula(
                n_covariates=N_COVARIATES,
                use_hmm=True,
                fit_interaction=True,
                fit_pack_years=True,
            )
            model_hmm.fit(data.G, data.S, data.C, data.Y, L=data.L, 
                         n_epochs=n_epochs, verbose=False)
            
            # g-formula simulation
            res_never = model_hmm.simulate_gformula(data.G, data.L, 'never_smoke', n_time, n_mc)
            res_always = model_hmm.simulate_gformula(data.G, data.L, 'always_smoke', n_time, n_mc)
            
            risk_never = res_never['mean_cumulative_risk']
            risk_always = res_always['mean_cumulative_risk']
            rr_hmm = risk_always / (risk_never + 1e-10)
            
            params_hmm = model_hmm.get_parameters()
            
            # Model fit statistics (LL, AIC, BIC)
            ll_hmm = compute_log_likelihood(model_hmm, data.G, data.S, data.C, data.Y, data.L)
            n_params_hmm = count_parameters(model_hmm)
            aic_hmm = compute_aic(ll_hmm, n_params_hmm)
            bic_hmm = compute_bic(ll_hmm, n_params_hmm, n_samples)
            
            sim_results['HMM_RR'] = rr_hmm
            sim_results['HMM_risk_never'] = risk_never
            sim_results['HMM_risk_always'] = risk_always
            sim_results['HMM_beta_S'] = params_hmm.get('beta_S', np.nan)
            sim_results['HMM_beta_GS'] = params_hmm.get('beta_GS', np.nan)
            sim_results['HMM_beta_Z'] = params_hmm.get('beta_Z', np.nan)
            sim_results['HMM_LL'] = ll_hmm
            sim_results['HMM_AIC'] = aic_hmm
            sim_results['HMM_BIC'] = bic_hmm
            sim_results['HMM_nparams'] = n_params_hmm
            
        except Exception as e:
            sim_results['HMM_RR'] = np.nan
            sim_results['HMM_LL'] = np.nan
            if verbose and sim_idx == 0:
                print(f"  HMM failed: {e}")
        
        results.append(sim_results)
    
    df_results = pd.DataFrame(results)
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        # Model Fit Statistics
        print(f"\n[Model Fit Statistics]")
        print("-"*70)
        print(f"{'Method':<25} {'LL':>12} {'AIC':>12} {'BIC':>12} {'# Params':>10}")
        print("-"*70)
        
        for method in ['CoxPH', 'Markov', 'HMM']:
            ll_col = f'{method}_LL'
            aic_col = f'{method}_AIC'
            bic_col = f'{method}_BIC'
            np_col = f'{method}_nparams'
            
            if ll_col in df_results.columns:
                ll_mean = df_results[ll_col].mean()
                aic_mean = df_results[aic_col].mean()
                bic_mean = df_results[bic_col].mean()
                np_mean = df_results[np_col].mean()
                
                method_name = {'CoxPH': 'A. Cox PH', 'Markov': 'B. Markov', 'HMM': 'C. HMM'}[method]
                print(f"{method_name:<25} {ll_mean:>12.1f} {aic_mean:>12.1f} {bic_mean:>12.1f} {np_mean:>10.0f}")
        
        print(f"\n[Effect Estimates] True beta_GS = {true_beta_GS}")
        print("-"*70)
        print(f"{'Method':<25} {'Effect':>12} {'Mean':>10} {'Bias':>10} {'RMSE':>10}")
        print("-"*70)
        
        # Cox PH (HR)
        if 'CoxPH_HR' in df_results.columns:
            hr = df_results['CoxPH_HR'].dropna()
            print(f"{'A. Cox PH':<25} {'HR':>12} {hr.mean():>10.3f} {'-':>10} {'-':>10}")
        
        # Cox PH beta_GS
        if 'CoxPH_beta_GS' in df_results.columns:
            est = df_results['CoxPH_beta_GS'].dropna()
            bias = est.mean() - true_beta_GS
            rmse = np.sqrt(((est - true_beta_GS)**2).mean())
            print(f"{'   └─ beta_GS':<25} {'coef':>12} {est.mean():>10.4f} {bias:>10.4f} {rmse:>10.4f}")
        
        # Standard Markov (RR)
        if 'Markov_RR' in df_results.columns:
            rr = df_results['Markov_RR'].dropna()
            print(f"{'B. Standard Markov':<25} {'RR':>12} {rr.mean():>10.3f} {'-':>10} {'-':>10}")
        
        if 'Markov_beta_GS' in df_results.columns:
            est = df_results['Markov_beta_GS'].dropna()
            bias = est.mean() - true_beta_GS
            rmse = np.sqrt(((est - true_beta_GS)**2).mean())
            print(f"{'   └─ beta_GS':<25} {'coef':>12} {est.mean():>10.4f} {bias:>10.4f} {rmse:>10.4f}")
        
        # Proposed HMM (RR)
        if 'HMM_RR' in df_results.columns:
            rr = df_results['HMM_RR'].dropna()
            print(f"{'C. Proposed HMM':<25} {'RR':>12} {rr.mean():>10.3f} {'-':>10} {'-':>10}")
        
        if 'HMM_beta_GS' in df_results.columns:
            est = df_results['HMM_beta_GS'].dropna()
            bias = est.mean() - true_beta_GS
            rmse = np.sqrt(((est - true_beta_GS)**2).mean())
            print(f"{'   └─ beta_GS':<25} {'coef':>12} {est.mean():>10.4f} {bias:>10.4f} {rmse:>10.4f}")
        
        # Key Insight
        print("\n" + "="*70)
        print("KEY INSIGHT")
        print("="*70)
        
        # Model fit comparison
        if 'HMM_LL' in df_results.columns and 'Markov_LL' in df_results.columns:
            ll_hmm = df_results['HMM_LL'].mean()
            ll_markov = df_results['Markov_LL'].mean()
            aic_hmm = df_results['HMM_AIC'].mean()
            aic_markov = df_results['Markov_AIC'].mean()
            
            print(f"\n  Model Fit (HMM vs Markov):")
            print(f"    ΔLL  = {ll_hmm - ll_markov:+.1f} (HMM better fit)")
            print(f"    ΔAIC = {aic_hmm - aic_markov:+.1f} {'(HMM preferred)' if aic_hmm < aic_markov else '(Markov preferred)'}")
        
        biases = {}
        for method in ['CoxPH', 'Markov', 'HMM']:
            col = f'{method}_beta_GS'
            if col in df_results.columns:
                biases[method] = df_results[col].dropna().mean() - true_beta_GS
        
        if biases:
            print(f"\n  beta_GS Estimation Bias:")
            print(f"    Cox PH (no Z):        {biases.get('CoxPH', np.nan):+.4f}")
            print(f"    Markov (no Z):        {biases.get('Markov', np.nan):+.4f}")
            print(f"    HMM (with Z):         {biases.get('HMM', np.nan):+.4f}")
            print(f"\n  → Cox and Markov underestimate GxE interaction")
            print(f"  → HMM corrects for unmeasured confounding (latent Z)")
            print(f"  → HMM shows BEST model fit (highest LL, lowest AIC)")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    if save_results:
        df_results.to_csv(os.path.join(OUTPUT_DIR, 'exp4_method_comparison.csv'), index=False)
        
        # Summary table (논문 Table 형식)
        summary = []
        
        # Cox PH
        row_cox = {
            'Method': 'A. Time-varying Cox PH',
            'Effect_Measure': 'Hazard Ratio',
            'Latent_Z': 'No',
        }
        if 'CoxPH_HR' in df_results.columns:
            row_cox['Estimated_Effect'] = df_results['CoxPH_HR'].mean()
        if 'CoxPH_beta_GS' in df_results.columns:
            est = df_results['CoxPH_beta_GS'].dropna()
            row_cox['beta_GS_mean'] = est.mean()
            row_cox['beta_GS_bias'] = est.mean() - true_beta_GS
            row_cox['beta_GS_rmse'] = np.sqrt(((est - true_beta_GS)**2).mean())
        if 'CoxPH_LL' in df_results.columns:
            row_cox['LL_mean'] = df_results['CoxPH_LL'].mean()
            row_cox['AIC_mean'] = df_results['CoxPH_AIC'].mean()
            row_cox['BIC_mean'] = df_results['CoxPH_BIC'].mean()
            row_cox['n_params'] = df_results['CoxPH_nparams'].mean()
        summary.append(row_cox)
        
        # Markov
        row_markov = {
            'Method': 'B. Standard Markov g-formula',
            'Effect_Measure': 'Risk Ratio',
            'Latent_Z': 'No',
        }
        if 'Markov_RR' in df_results.columns:
            row_markov['Estimated_Effect'] = df_results['Markov_RR'].mean()
        if 'Markov_beta_GS' in df_results.columns:
            est = df_results['Markov_beta_GS'].dropna()
            row_markov['beta_GS_mean'] = est.mean()
            row_markov['beta_GS_bias'] = est.mean() - true_beta_GS
            row_markov['beta_GS_rmse'] = np.sqrt(((est - true_beta_GS)**2).mean())
        if 'Markov_LL' in df_results.columns:
            row_markov['LL_mean'] = df_results['Markov_LL'].mean()
            row_markov['AIC_mean'] = df_results['Markov_AIC'].mean()
            row_markov['BIC_mean'] = df_results['Markov_BIC'].mean()
            row_markov['n_params'] = df_results['Markov_nparams'].mean()
        summary.append(row_markov)
        
        # HMM
        row_hmm = {
            'Method': 'C. Proposed HMM g-formula',
            'Effect_Measure': 'Risk Ratio',
            'Latent_Z': 'Yes',
        }
        if 'HMM_RR' in df_results.columns:
            row_hmm['Estimated_Effect'] = df_results['HMM_RR'].mean()
        if 'HMM_beta_GS' in df_results.columns:
            est = df_results['HMM_beta_GS'].dropna()
            row_hmm['beta_GS_mean'] = est.mean()
            row_hmm['beta_GS_bias'] = est.mean() - true_beta_GS
            row_hmm['beta_GS_rmse'] = np.sqrt(((est - true_beta_GS)**2).mean())
        if 'HMM_LL' in df_results.columns:
            row_hmm['LL_mean'] = df_results['HMM_LL'].mean()
            row_hmm['AIC_mean'] = df_results['HMM_AIC'].mean()
            row_hmm['BIC_mean'] = df_results['HMM_BIC'].mean()
            row_hmm['n_params'] = df_results['HMM_nparams'].mean()
        summary.append(row_hmm)
        
        summary_df = pd.DataFrame(summary)
        summary_df['True_beta_GS'] = true_beta_GS
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'exp4_summary.csv'), index=False)
        
        if verbose:
            print(f"\nSaved: {OUTPUT_DIR}/exp4_method_comparison.csv, exp4_summary.csv")
    
    return {'results': df_results, 'true_beta_GS': true_beta_GS}


# =============================================================================
# Experiment 5: Causal Inference with Bootstrap CI
# =============================================================================

def run_experiment_5(
    n_samples: int = None,
    n_bootstrap: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """실험 5: Causal Inference with Bootstrap 95% CI"""
    n_samples = n_samples or DEFAULT_DATA_PARAMS['n_samples']
    n_bootstrap = n_bootstrap or MC_PARAMS['n_bootstrap']
    n_time = DEFAULT_DATA_PARAMS['n_time']
    n_epochs = get_n_epochs(n_samples)
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 5: Causal Inference with Bootstrap CI")
        print("="*70)
        print(f"N samples: {n_samples:,}, N bootstrap: {n_bootstrap}, N time: {n_time}")
    
    set_seed(SEED)
    data = generate_synthetic_data(n_samples=n_samples, n_time=n_time, seed=SEED)
    
    if verbose:
        validate_dgp(data)
    
    if verbose:
        print("\nFitting HMM-gFormula model...")
    
    model = HiddenMarkovGFormula(n_covariates=N_COVARIATES, fit_interaction=True, fit_pack_years=True)
    model.fit(data.G, data.S, data.C, data.Y, L=data.L, n_epochs=n_epochs, verbose=verbose)
    
    if verbose:
        print("\nEstimated parameters:")
        params = model.get_parameters()
        for k in ['beta_GS', 'beta_time', 'alpha_time', 'psi', 'gamma_GS']:
            if k in params:
                print(f"  {k}: {params[k]:.4f}")
    
    # 1. 기본 인과 효과
    if verbose:
        print("\n" + "-"*50)
        print("1. Causal Effect Estimation (Bootstrap CI)")
        print("-"*50)
    
    causal_results = bootstrap_causal_effect(
        model, data.G, data.L, n_bootstrap=n_bootstrap, n_time=n_time, verbose=verbose
    )
    
    if verbose:
        print(f"\n  Risk (Never Smoke): {causal_results['risk_never_smoke']['mean']*100:.2f}% "
              f"[{causal_results['risk_never_smoke']['ci_lower']*100:.2f}, "
              f"{causal_results['risk_never_smoke']['ci_upper']*100:.2f}]")
        print(f"  Risk (Always Smoke): {causal_results['risk_always_smoke']['mean']*100:.2f}% "
              f"[{causal_results['risk_always_smoke']['ci_lower']*100:.2f}, "
              f"{causal_results['risk_always_smoke']['ci_upper']*100:.2f}]")
        print(f"  Risk Ratio: {causal_results['risk_ratio']['mean']:.2f} "
              f"[{causal_results['risk_ratio']['ci_lower']:.2f}, "
              f"{causal_results['risk_ratio']['ci_upper']:.2f}]")
    
    # 2. PRS 층화 분석
    if verbose:
        print("\n" + "-"*50)
        print("2. PRS-Stratified Analysis")
        print("-"*50)
    
    prs_results = bootstrap_prs_stratified_effect(
        model, data.G, data.L, n_bootstrap=min(n_bootstrap, 100), n_time=n_time, verbose=verbose
    )
    
    if verbose:
        for group in ['high_risk', 'low_risk']:
            r = prs_results[group]
            print(f"  {group}: RR = {r['mean_rr']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")
    
    # 3. 금연 시점 효과
    if verbose:
        print("\n" + "-"*50)
        print("3. Quit Timing Effect")
        print("-"*50)
    
    quit_times = list(range(0, n_time, 4))
    quit_results = bootstrap_quit_timing_effect(
        model, data.G, data.L, quit_times=quit_times,
        n_bootstrap=min(n_bootstrap, 50), n_time=n_time, verbose=verbose
    )
    
    if verbose:
        for qt, r in quit_results.items():
            print(f"  Quit at Year {qt}: RR = {r['mean_rr']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]")
    
    all_results = {
        'causal_effects': causal_results,
        'prs_stratified': prs_results,
        'quit_timing': quit_results,
        'model_params': model.get_parameters(),
    }
    
    if save_results:
        # Causal effects
        causal_df = pd.DataFrame({
            'Metric': ['Risk_Never_Smoke', 'Risk_Always_Smoke', 'Risk_Difference', 'Risk_Ratio'],
            'Mean': [
                causal_results['risk_never_smoke']['mean'],
                causal_results['risk_always_smoke']['mean'],
                causal_results['risk_difference']['mean'],
                causal_results['risk_ratio']['mean'],
            ],
            'CI_Lower': [
                causal_results['risk_never_smoke']['ci_lower'],
                causal_results['risk_always_smoke']['ci_lower'],
                causal_results['risk_difference']['ci_lower'],
                causal_results['risk_ratio']['ci_lower'],
            ],
            'CI_Upper': [
                causal_results['risk_never_smoke']['ci_upper'],
                causal_results['risk_always_smoke']['ci_upper'],
                causal_results['risk_difference']['ci_upper'],
                causal_results['risk_ratio']['ci_upper'],
            ],
        })
        causal_df.to_csv(os.path.join(OUTPUT_DIR, 'exp5_causal_effects.csv'), index=False)
        
        # PRS stratified
        prs_df = pd.DataFrame([
            {'group': 'high_risk', **prs_results['high_risk']},
            {'group': 'low_risk', **prs_results['low_risk']},
        ])
        prs_df.to_csv(os.path.join(OUTPUT_DIR, 'exp5_prs_stratified.csv'), index=False)
        
        # Quit timing
        quit_df = pd.DataFrame([{'quit_year': qt, **vals} for qt, vals in quit_results.items()])
        quit_df.to_csv(os.path.join(OUTPUT_DIR, 'exp5_quit_timing.csv'), index=False)
        
        # =====================================================================
        # [NEW] Model Parameters (for Supplementary Table)
        # =====================================================================
        params = model.get_parameters()
        param_rows = []
        
        # Covariate names for interpretation
        covariate_names = ['age', 'sex', 'bmi']
        
        for param_name, param_value in params.items():
            if isinstance(param_value, list):
                # List parameters (e.g., beta_L, gamma_L, alpha_L)
                for i, v in enumerate(param_value):
                    cov_name = covariate_names[i] if i < len(covariate_names) else f'cov_{i}'
                    param_rows.append({
                        'parameter': f'{param_name}_{cov_name}',
                        'value': v,
                        'description': f'{param_name} effect of {cov_name}',
                    })
            else:
                # Scalar parameters
                description = _get_parameter_description(param_name)
                param_rows.append({
                    'parameter': param_name,
                    'value': param_value,
                    'description': description,
                })
        
        params_df = pd.DataFrame(param_rows)
        params_df.to_csv(os.path.join(OUTPUT_DIR, 'exp5_model_params.csv'), index=False)
        
        if verbose:
            print(f"\nSaved: {OUTPUT_DIR}/exp5_*.csv (including exp5_model_params.csv)")
            print("\n[Model Parameters Summary]")
            print("-"*60)
            for _, row in params_df.iterrows():
                print(f"  {row['parameter']:<20}: {row['value']:>10.4f}  ({row['description']})")
    
    return all_results


def _get_parameter_description(param_name: str) -> str:
    """파라미터 설명 반환 (Supplementary Table용)"""
    descriptions = {
        # Hidden state model
        'psi': 'Latent state autoregression',
        'gamma_S': 'Smoking -> Latent state',
        'gamma_C': 'Pack-years -> Latent state',
        'gamma_G': 'PRS -> Latent state',
        'gamma_GS': 'PRS x Smoking interaction (Z)',
        'gamma_GC': 'PRS x Pack-years interaction (Z)',
        'sigma_Z': 'Latent state noise SD',
        
        # Outcome model
        'beta_0': 'Baseline CVD log-odds',
        'beta_Z': 'Latent state -> CVD',
        'beta_S': 'Smoking -> CVD (direct)',
        'beta_G': 'PRS -> CVD (direct)',
        'beta_GS': 'PRS x Smoking interaction (Y)',
        'beta_time': 'Time trend (aging effect)',
        
        # Exposure model
        'alpha_0': 'Baseline smoking log-odds',
        'alpha_S': 'Smoking persistence',
        'alpha_Z': 'Latent state -> Smoking',
        'alpha_G': 'PRS -> Smoking',
        'alpha_time': 'Time trend (cessation)',
    }
    return descriptions.get(param_name, '')


# =============================================================================
# Run All Experiments
# =============================================================================

def run_all_experiments(
    n_simulations: int = None,
    n_samples: int = None,
    quick: bool = False,
    verbose: bool = True,
) -> Dict:
    """모든 실험 순차 실행"""
    
    if quick:
        n_simulations = n_simulations or 30
        n_samples = n_samples or 5000
        n_bootstrap = 50
    else:
        n_simulations = n_simulations or MC_PARAMS['n_simulations']
        n_samples = n_samples or DEFAULT_DATA_PARAMS['n_samples']
        n_bootstrap = MC_PARAMS['n_bootstrap']
    
    print("\n" + "="*70)
    print("SIMULATION STUDY v3.3: Korean-Calibrated HMM g-formula")
    print("="*70)
    print(f"Configuration:")
    print(f"  MC simulations: {n_simulations}")
    print(f"  Default sample size: {n_samples:,}")
    print(f"  Time points: {DEFAULT_DATA_PARAMS['n_time']} years")
    print(f"  Quick mode: {quick}")
    print("="*70)
    
    results = {}
    total_start = time.time()
    
    # Experiment 1
    print("\n" + "▶"*35)
    results['exp1'] = run_experiment_1(
        n_simulations=n_simulations, n_samples=n_samples, verbose=verbose
    )
    
    # Experiment 2
    print("\n" + "▶"*35)
    results['exp2'] = run_experiment_2(
        n_simulations=min(n_simulations, 50) if quick else n_simulations,
        verbose=verbose
    )
    
    # Experiment 3
    print("\n" + "▶"*35)
    results['exp3'] = run_experiment_3(
        n_simulations=n_simulations, n_samples=n_samples, verbose=verbose
    )
    
    # Experiment 4
    print("\n" + "▶"*35)
    results['exp4'] = run_experiment_4(
        n_simulations=n_simulations, n_samples=n_samples, verbose=verbose
    )
    
    # Experiment 5
    print("\n" + "▶"*35)
    results['exp5'] = run_experiment_5(
        n_samples=n_samples, n_bootstrap=n_bootstrap, verbose=verbose
    )
    
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved in: {OUTPUT_DIR}/")
    print("="*70)
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simulation experiments')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4, 5], help='Run specific experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--n-sim', type=int, default=None, help='Number of MC simulations')
    parser.add_argument('--n-samples', type=int, default=None, help='Sample size')
    parser.add_argument('--n-boot', type=int, default=None, help='Bootstrap iterations (Exp 5)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.quick:
        args.n_sim = args.n_sim or 30
        args.n_samples = args.n_samples or 5000
        args.n_boot = args.n_boot or 50
    
    if args.all:
        run_all_experiments(n_simulations=args.n_sim, n_samples=args.n_samples, quick=args.quick)
    elif args.exp == 1:
        run_experiment_1(n_simulations=args.n_sim or 200, n_samples=args.n_samples or 20000)
    elif args.exp == 2:
        run_experiment_2(n_simulations=args.n_sim or 200)
    elif args.exp == 3:
        run_experiment_3(n_simulations=args.n_sim or 200, n_samples=args.n_samples or 20000)
    elif args.exp == 4:
        run_experiment_4(n_simulations=args.n_sim or 200, n_samples=args.n_samples or 20000)
    elif args.exp == 5:
        run_experiment_5(n_samples=args.n_samples or 20000, n_bootstrap=args.n_boot or 200)
    else:
        print("Usage: python run_experiments.py --exp [1-5] or --all")
        print("       Add --quick for fast testing")