"""
experiments/run_experiments.py - Main Experiment Runner (v3.1)

일관된 MC 횟수, 대규모 sample size 지원

실험 1: Effect Size별 Parameter Recovery
실험 2: Sample Size별 추정 안정성 (5K ~ 100K)
실험 3: Model Misspecification
실험 4: Method Comparison
실험 5: Causal Inference with Bootstrap CI
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
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
from data_generator import generate_synthetic_data, validate_dgp
from models import HiddenMarkovGFormula, estimate_causal_effect

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Bootstrap Utilities
# =============================================================================

def bootstrap_causal_effect(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    L: torch.Tensor,
    n_bootstrap: int = 200,
    n_time: int = 10,
    confidence_level: float = 0.95,
    verbose: bool = True,
) -> Dict:
    """
    Bootstrap을 이용한 인과 효과 추정 및 95% CI 계산
    """
    n_samples = G.shape[0]
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    
    risk_never_list = []
    risk_always_list = []
    risk_diff_list = []
    risk_ratio_list = []
    
    iterator = tqdm(range(n_bootstrap), desc="Bootstrap") if verbose else range(n_bootstrap)
    
    for b in iterator:
        # Bootstrap 샘플링 (replacement)
        idx = torch.randint(0, n_samples, (n_samples,))
        G_boot = G[idx]
        L_boot = L[idx]
        
        # 각 intervention에 대한 risk 계산
        result_never = model.simulate_gformula(G_boot, L_boot, 'never_smoke', n_time, n_monte_carlo=n_mc)
        result_always = model.simulate_gformula(G_boot, L_boot, 'always_smoke', n_time, n_monte_carlo=n_mc)
        
        risk_never = result_never['mean_cumulative_risk']
        risk_always = result_always['mean_cumulative_risk']
        
        risk_never_list.append(risk_never)
        risk_always_list.append(risk_always)
        risk_diff_list.append(risk_always - risk_never)
        risk_ratio_list.append(risk_always / (risk_never + 1e-10))
    
    # 통계량 계산
    alpha = 1 - confidence_level
    
    def compute_ci(values):
        values = np.array(values)
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, 100 * alpha / 2),
            'ci_upper': np.percentile(values, 100 * (1 - alpha / 2)),
        }
    
    results = {
        'risk_never_smoke': compute_ci(risk_never_list),
        'risk_always_smoke': compute_ci(risk_always_list),
        'risk_difference': compute_ci(risk_diff_list),
        'risk_ratio': compute_ci(risk_ratio_list),
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
    }
    
    return results


def bootstrap_quit_timing_effect(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    L: torch.Tensor,
    quit_times: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    n_bootstrap: int = 100,
    n_time: int = 10,
    verbose: bool = True,
) -> Dict:
    """금연 시점에 따른 효과 변화 (Bootstrap CI 포함)"""
    results = {t: [] for t in quit_times}
    n_samples = G.shape[0]
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap quit timing", disable=not verbose):
        idx = torch.randint(0, n_samples, (n_samples,))
        G_boot = G[idx]
        L_boot = L[idx]
        
        ref = model.simulate_gformula(G_boot, L_boot, 'always_smoke', n_time, n_monte_carlo=n_mc)
        risk_always = ref['mean_cumulative_risk']
        
        for qt in quit_times:
            res = model.simulate_gformula(G_boot, L_boot, f'quit_at_t{qt}', n_time, n_monte_carlo=n_mc, quit_time=qt)
            risk_quit = res['mean_cumulative_risk']
            rr = risk_quit / (risk_always + 1e-10)
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
    n_time: int = 10,
    prs_percentile: float = 80,
    verbose: bool = True,
) -> Dict:
    """PRS 층화 분석: High Risk (상위 20%) vs Low Risk (하위 80%)"""
    n_samples = G.shape[0]
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    
    prs_threshold = torch.quantile(G, prs_percentile / 100)
    high_risk_mask = G.squeeze() >= prs_threshold
    low_risk_mask = ~high_risk_mask
    
    results = {'high_risk': [], 'low_risk': []}
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap PRS stratified", disable=not verbose):
        for group, mask in [('high_risk', high_risk_mask), ('low_risk', low_risk_mask)]:
            G_group = G[mask]
            L_group = L[mask]
            
            if len(G_group) < 10:
                continue
            
            n_group = len(G_group)
            idx = torch.randint(0, n_group, (n_group,))
            G_boot = G_group[idx]
            L_boot = L_group[idx]
            
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
            'n_bootstrap': len(values),
        }
    
    return summary


# =============================================================================
# Experiment 1: Effect Size Sensitivity
# =============================================================================

def run_experiment_1(
    n_simulations: int = None,
    n_samples: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """실험 1: Effect size에 따른 Parameter Recovery 분석"""
    n_sim = n_simulations or MC_PARAMS['n_simulations']
    n_samples = n_samples or DEFAULT_DATA_PARAMS['n_samples']
    n_time = DEFAULT_DATA_PARAMS['n_time']
    n_epochs = get_n_epochs(n_samples)
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 1: Effect Size Sensitivity Analysis")
        print("="*70)
        print(f"N simulations: {n_sim}, N samples: {n_samples}, Epochs: {n_epochs}")
    
    results = {scenario: defaultdict(list) for scenario in EXP1_EFFECT_SIZES}
    true_values = {}
    
    for scenario_name, effect_mods in EXP1_EFFECT_SIZES.items():
        if verbose:
            print(f"\n>>> Processing: {scenario_name}")
        
        params_h = get_modified_params(TRUE_PARAMS_HIDDEN_STATE, {
            'gamma_GS': effect_mods['gamma_GS'], 
            'gamma_GC': effect_mods['gamma_GC']
        })
        params_o = get_modified_params(TRUE_PARAMS_OUTCOME, {
            'beta_GS': effect_mods['beta_GS']
        })
        true_values[scenario_name] = effect_mods['beta_GS']
        
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
        print("Results Summary (beta_GS):")
        print("-"*70)
        for scenario, true_val in true_values.items():
            if results[scenario]['beta_GS']:
                est = np.array(results[scenario]['beta_GS'])
                bias = np.mean(est) - true_val
                rmse = np.sqrt(np.mean((est - true_val)**2))
                print(f"{scenario}: True={true_val:.3f}, Est={np.mean(est):.3f}, "
                      f"Bias={bias:.3f}, RMSE={rmse:.3f}")
    
    if save_results:
        df = []
        for scenario, param_dict in results.items():
            for param, values in param_dict.items():
                for v in values:
                    df.append({'scenario': scenario, 'parameter': param, 'estimate': v})
        pd.DataFrame(df).to_csv(os.path.join(OUTPUT_DIR, 'exp1_results.csv'), index=False)
    
    return {'results': {k: dict(v) for k, v in results.items()}, 'true_values': true_values}


# =============================================================================
# Experiment 2: Sample Size Robustness
# =============================================================================

def run_experiment_2(
    n_simulations: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    실험 2: Sample size에 따른 추정 안정성 분석
    
    N = [5000, 10000, 20000, 50000, 100000]
    CVD rare outcome (~3-4%)을 고려한 대규모 sample size 평가
    """
    n_sim = n_simulations or MC_PARAMS['n_simulations']
    n_time = DEFAULT_DATA_PARAMS['n_time']
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 2: Sample Size Robustness Analysis")
        print("="*70)
        print(f"N simulations: {n_sim}")
        print(f"Sample sizes: {EXP2_SAMPLE_SIZES}")
        print("(Using consistent MC iterations across all sample sizes)")
    
    # Use moderate effect
    effect_mods = EXP1_EFFECT_SIZES['Moderate']
    params_h = get_modified_params(TRUE_PARAMS_HIDDEN_STATE, {
        'gamma_GS': effect_mods['gamma_GS'], 
        'gamma_GC': effect_mods['gamma_GC']
    })
    params_o = get_modified_params(TRUE_PARAMS_OUTCOME, {
        'beta_GS': effect_mods['beta_GS']
    })
    true_beta_GS = effect_mods['beta_GS']
    
    results = {n: defaultdict(list) for n in EXP2_SAMPLE_SIZES}
    timing = {}
    
    for n_samples in EXP2_SAMPLE_SIZES:
        n_epochs = get_n_epochs(n_samples)
        
        if verbose:
            print(f"\n>>> Processing N = {n_samples:,} (epochs={n_epochs})")
        
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
        
        elapsed = time.time() - start_time
        timing[n_samples] = elapsed
        
        if verbose:
            print(f"    Completed in {elapsed/60:.1f} minutes")
    
    # Summary
    if verbose:
        print("\n" + "-"*70)
        print(f"Results Summary (beta_GS, True = {true_beta_GS})")
        print("-"*70)
        print(f"{'N':>10} {'Mean':>10} {'Bias':>10} {'RMSE':>10} {'Coverage':>10} {'Time(min)':>10}")
        print("-"*70)
        
        for n_samples in EXP2_SAMPLE_SIZES:
            if results[n_samples]['beta_GS']:
                est = np.array(results[n_samples]['beta_GS'])
                bias = np.mean(est) - true_beta_GS
                rmse = np.sqrt(np.mean((est - true_beta_GS)**2))
                # Approximate coverage
                ci_lower = np.percentile(est, 2.5)
                ci_upper = np.percentile(est, 97.5)
                coverage = 1.0 if ci_lower <= true_beta_GS <= ci_upper else 0.0
                
                print(f"{n_samples:>10,} {np.mean(est):>10.4f} {bias:>10.4f} "
                      f"{rmse:>10.4f} {coverage:>10.2f} {timing[n_samples]/60:>10.1f}")
    
    if save_results:
        # Results CSV
        df = []
        for n_samples, param_dict in results.items():
            for param, values in param_dict.items():
                for v in values:
                    df.append({'sample_size': n_samples, 'parameter': param, 'estimate': v})
        pd.DataFrame(df).to_csv(os.path.join(OUTPUT_DIR, 'exp2_results.csv'), index=False)
        
        # Summary CSV
        summary_rows = []
        for n_samples in EXP2_SAMPLE_SIZES:
            if results[n_samples]['beta_GS']:
                est = np.array(results[n_samples]['beta_GS'])
                summary_rows.append({
                    'n_samples': n_samples,
                    'true_value': true_beta_GS,
                    'mean_estimate': np.mean(est),
                    'bias': np.mean(est) - true_beta_GS,
                    'rmse': np.sqrt(np.mean((est - true_beta_GS)**2)),
                    'std': np.std(est),
                    'time_minutes': timing[n_samples] / 60,
                })
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'exp2_summary.csv'), index=False
        )
        
        if verbose:
            print(f"\nResults saved to {OUTPUT_DIR}/")
    
    return {
        'results': {k: dict(v) for k, v in results.items()},
        'true_beta_GS': true_beta_GS,
        'timing': timing,
    }


# =============================================================================
# Experiment 5: Causal Inference with Bootstrap CI
# =============================================================================

def run_experiment_5(
    n_samples: int = None,
    n_bootstrap: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    실험 5: Causal Inference with Bootstrap 95% CI
    
    - Risk Ratio, Risk Difference의 점추정 및 95% CI
    - PRS 층화 분석 (High vs Low risk)
    - 금연 시점 효과 분석
    """
    n_samples = n_samples or DEFAULT_DATA_PARAMS['n_samples']
    n_bootstrap = n_bootstrap or MC_PARAMS['n_bootstrap']
    n_time = DEFAULT_DATA_PARAMS['n_time']
    n_epochs = get_n_epochs(n_samples)
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 5: Causal Inference with Bootstrap 95% CI")
        print("="*70)
        print(f"N samples: {n_samples:,}, N bootstrap: {n_bootstrap}, Epochs: {n_epochs}")
        print(f"g-formula MC per subject: {GFORMULA_PARAMS['n_monte_carlo']}")
    
    # 데이터 생성
    set_seed(SEED)
    data = generate_synthetic_data(n_samples=n_samples, n_time=n_time, seed=SEED)
    
    if verbose:
        validate_dgp(data)
    
    # 모델 학습
    if verbose:
        print("\nFitting HMM-gFormula model...")
    
    model = HiddenMarkovGFormula(n_covariates=N_COVARIATES, fit_interaction=True, fit_pack_years=True)
    model.fit(data.G, data.S, data.C, data.Y, L=data.L, n_epochs=n_epochs, verbose=verbose)
    
    if verbose:
        print("\nEstimated parameters:")
        params = model.get_parameters()
        for k, v in list(params.items())[:10]:
            if isinstance(v, list):
                print(f"  {k}: {[f'{x:.3f}' for x in v]}")
            else:
                print(f"  {k}: {v:.4f}")
    
    # =================================================================
    # 1. 기본 인과 효과 추정 (Bootstrap CI)
    # =================================================================
    if verbose:
        print("\n" + "-"*50)
        print("1. Causal Effect Estimation (Bootstrap CI)")
        print("-"*50)
    
    causal_results = bootstrap_causal_effect(
        model, data.G, data.L, n_bootstrap=n_bootstrap, n_time=n_time, verbose=verbose
    )
    
    if verbose:
        print("\nResults:")
        print(f"\n  Risk (Never Smoke):")
        print(f"    Mean: {causal_results['risk_never_smoke']['mean']*100:.2f}%")
        print(f"    95% CI: [{causal_results['risk_never_smoke']['ci_lower']*100:.2f}%, "
              f"{causal_results['risk_never_smoke']['ci_upper']*100:.2f}%]")
        
        print(f"\n  Risk (Always Smoke):")
        print(f"    Mean: {causal_results['risk_always_smoke']['mean']*100:.2f}%")
        print(f"    95% CI: [{causal_results['risk_always_smoke']['ci_lower']*100:.2f}%, "
              f"{causal_results['risk_always_smoke']['ci_upper']*100:.2f}%]")
        
        print(f"\n  Risk Difference (Smoke - No Smoke):")
        print(f"    Mean: {causal_results['risk_difference']['mean']*100:.2f}%")
        print(f"    95% CI: [{causal_results['risk_difference']['ci_lower']*100:.2f}%, "
              f"{causal_results['risk_difference']['ci_upper']*100:.2f}%]")
        
        print(f"\n  Risk Ratio (Smoke / No Smoke):")
        print(f"    Mean: {causal_results['risk_ratio']['mean']:.2f}")
        print(f"    95% CI: [{causal_results['risk_ratio']['ci_lower']:.2f}, "
              f"{causal_results['risk_ratio']['ci_upper']:.2f}]")
    
    # =================================================================
    # 2. PRS 층화 분석
    # =================================================================
    if verbose:
        print("\n" + "-"*50)
        print("2. PRS-Stratified Analysis (High vs Low Risk)")
        print("-"*50)
    
    prs_results = bootstrap_prs_stratified_effect(
        model, data.G, data.L, n_bootstrap=min(n_bootstrap, 100), n_time=n_time, verbose=verbose
    )
    
    if verbose:
        print("\nRisk Ratio (Smoke vs No Smoke) by PRS Group:")
        for group in ['high_risk', 'low_risk']:
            r = prs_results[group]
            label = 'top 20%' if group == 'high_risk' else 'bottom 80%'
            print(f"\n  {group.replace('_', ' ').title()} (PRS {label}):")
            print(f"    RR Mean: {r['mean_rr']:.2f}")
            print(f"    95% CI: [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")
    
    # =================================================================
    # 3. 금연 시점 효과 분석
    # =================================================================
    if verbose:
        print("\n" + "-"*50)
        print("3. Effect of Quit Timing")
        print("-"*50)
    
    quit_results = bootstrap_quit_timing_effect(
        model, data.G, data.L, quit_times=[0, 2, 4, 6, 8],
        n_bootstrap=min(n_bootstrap, 50), n_time=n_time, verbose=verbose
    )
    
    if verbose:
        print("\nRisk Ratio (Quit at Year t vs Always Smoke):")
        for qt, r in quit_results.items():
            print(f"  Quit at Year {qt}: RR = {r['mean_rr']:.3f} "
                  f"(95% CI: [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}])")
    
    # 결과 저장
    all_results = {
        'causal_effects': causal_results,
        'prs_stratified': prs_results,
        'quit_timing': quit_results,
        'model_params': model.get_parameters(),
    }
    
    if save_results:
        summary_df = pd.DataFrame({
            'Metric': ['Risk (Never Smoke)', 'Risk (Always Smoke)', 'Risk Difference', 'Risk Ratio'],
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
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'exp5_causal_effects.csv'), index=False)
        
        if verbose:
            print(f"\nResults saved to {OUTPUT_DIR}/")
    
    return all_results


# =============================================================================
# Run All Experiments
# =============================================================================

def run_all_experiments(
    n_simulations: int = None,
    quick: bool = False,
    verbose: bool = True,
) -> Dict:
    """모든 실험 실행"""
    n_sim = MC_PARAMS['n_simulations_quick'] if quick else (n_simulations or MC_PARAMS['n_simulations'])
    
    print("\n" + "="*70)
    print("SIMULATION STUDY v3.1: Korean-Calibrated HMM g-formula")
    print("="*70)
    print(f"Monte Carlo simulations: {n_sim}")
    print(f"Sample sizes for Exp2: {EXP2_SAMPLE_SIZES}")
    print(f"Quick mode: {quick}")
    
    results = {}
    
    # Experiment 1: Effect Size
    results['exp1'] = run_experiment_1(
        n_simulations=n_sim,
        n_samples=20000 if not quick else 5000,
        verbose=verbose
    )
    
    # Experiment 2: Sample Size (skip largest if quick)
    if quick:
        # Quick mode: only test smaller sizes
        original_sizes = EXP2_SAMPLE_SIZES.copy()
        results['exp2'] = run_experiment_2(
            n_simulations=min(n_sim, 30),
            verbose=verbose
        )
    else:
        results['exp2'] = run_experiment_2(
            n_simulations=n_sim,
            verbose=verbose
        )
    
    # Experiment 5: Causal Inference
    results['exp5'] = run_experiment_5(
        n_samples=20000 if not quick else 5000,
        n_bootstrap=MC_PARAMS['n_bootstrap'] if not quick else 50,
        verbose=verbose
    )
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simulation experiments')
    parser.add_argument('--exp', type=int, choices=[1, 2, 5], help='Run specific experiment')
    parser.add_argument('--n-sim', type=int, default=None, help='Number of simulations')
    parser.add_argument('--n-samples', type=int, default=None, help='Sample size')
    parser.add_argument('--n-boot', type=int, default=None, help='Bootstrap iterations')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.exp == 1:
        run_experiment_1(
            n_simulations=args.n_sim or (30 if args.quick else 200),
            n_samples=args.n_samples or (5000 if args.quick else 20000),
        )
    elif args.exp == 2:
        run_experiment_2(
            n_simulations=args.n_sim or (30 if args.quick else 200),
        )
    elif args.exp == 5:
        run_experiment_5(
            n_samples=args.n_samples or (5000 if args.quick else 20000),
            n_bootstrap=args.n_boot or (50 if args.quick else 200),
        )
    else:
        run_all_experiments(
            n_simulations=args.n_sim,
            quick=args.quick,
        )