"""
analysis_advanced_prs.py - Advanced Analysis with 95% CI Shaded Spline Curves

v3.3 Changes:
- [UPDATE] n_time 기본값 10 → 20 (20년 시뮬레이션)
- [UPDATE] quit_years 동적 생성 (2년 간격)
- [UPDATE] x축 범위 데이터 기반 동적 설정

v3.2 Changes:
- [NEW] Bootstrap CI 음영 처리 (plt.fill_between)
- [NEW] 상한선/하한선 Spline 보간

분석 내용:
1. Curve A (Effect Modification): PRS vs Risk Ratio - 유전적 위험도에 따른 흡연 효과
2. Curve B (Urgency of Cessation): Quit Year vs Risk Ratio - 금연 지연에 따른 이득 감소

시각화:
- 평균선: Cubic Spline
- 95% CI: fill_between으로 음영 처리
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, CubicSpline
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SEED, DEFAULT_DATA_PARAMS, OUTPUT_DIR, N_COVARIATES, MC_PARAMS, GFORMULA_PARAMS
from data_generator import generate_synthetic_data, validate_dgp
from models import HiddenMarkovGFormula

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Analysis A: PRS vs Risk Ratio with Bootstrap CI
# =============================================================================

def analyze_prs_effect_with_bootstrap(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    L: torch.Tensor,
    n_prs_points: int = 15,
    n_bootstrap: int = 100,
    n_time: int = 20,  # 20년 시뮬레이션
    verbose: bool = True,
) -> pd.DataFrame:
    """
    PRS별 Risk Ratio 계산 + Bootstrap CI
    
    Returns:
        DataFrame with columns: prs, rr_mean, rr_lower, rr_upper
    """
    if verbose:
        print("Analyzing PRS effect modification with bootstrap CI...")
    
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    
    # PRS 구간 설정
    prs_min, prs_max = G.min().item(), G.max().item()
    prs_points = np.linspace(prs_min * 0.9, prs_max * 0.9, n_prs_points)
    bandwidth = G.std().item() * 0.4
    
    results = []
    
    for prs_val in tqdm(prs_points, desc="PRS points", disable=not verbose):
        # 해당 PRS 근처 샘플 선택
        mask = (torch.abs(G.squeeze() - prs_val) < bandwidth)
        
        if mask.sum() < 50:
            continue
        
        G_subset = G[mask]
        L_subset = L[mask]
        n_subset = len(G_subset)
        
        # Bootstrap
        rr_bootstrap = []
        
        for b in range(n_bootstrap):
            idx = torch.randint(0, n_subset, (n_subset,))
            G_boot = G_subset[idx]
            L_boot = L_subset[idx]
            
            res_never = model.simulate_gformula(G_boot, L_boot, 'never_smoke', n_time, n_mc // 2)
            res_always = model.simulate_gformula(G_boot, L_boot, 'always_smoke', n_time, n_mc // 2)
            
            rr = res_always['mean_cumulative_risk'] / (res_never['mean_cumulative_risk'] + 1e-10)
            rr_bootstrap.append(rr)
        
        rr_array = np.array(rr_bootstrap)
        
        results.append({
            'prs': prs_val,
            'rr_mean': np.mean(rr_array),
            'rr_lower': np.percentile(rr_array, 2.5),
            'rr_upper': np.percentile(rr_array, 97.5),
            'rr_std': np.std(rr_array),
        })
    
    return pd.DataFrame(results)


def analyze_quit_timing_with_bootstrap(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    L: torch.Tensor,
    quit_years: List[int] = None,  # None이면 n_time 기반으로 자동 생성
    n_bootstrap: int = 100,
    n_time: int = 20,  # 20년 시뮬레이션
    prs_groups: bool = True,
    prs_percentile: float = 80,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    금연 시점별 Risk Ratio + Bootstrap CI
    
    Returns:
        DataFrame with columns: quit_year, group, rr_mean, rr_lower, rr_upper
    """
    if verbose:
        print("Analyzing quit timing effect with bootstrap CI...")
    
    # quit_years 기본값: 2년 간격으로 0부터 n_time까지
    if quit_years is None:
        quit_years = list(range(0, n_time + 1, 2))
    
    n_mc = GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    n_samples = G.shape[0]
    
    # PRS 층화
    prs_threshold = torch.quantile(G, prs_percentile / 100)
    
    groups = [('All', torch.ones(n_samples, dtype=torch.bool))]
    if prs_groups:
        groups.append(('High_PRS', G.squeeze() >= prs_threshold))
        groups.append(('Low_PRS', G.squeeze() < prs_threshold))
    
    results = []
    
    for group_name, group_mask in groups:
        G_group = G[group_mask]
        L_group = L[group_mask]
        n_group = len(G_group)
        
        if n_group < 100:
            continue
        
        if verbose:
            print(f"  Processing group: {group_name} (n={n_group})")
        
        for qt in tqdm(quit_years, desc=f"  {group_name}", disable=not verbose):
            rr_bootstrap = []
            
            for b in range(n_bootstrap):
                idx = torch.randint(0, n_group, (n_group,))
                G_boot = G_group[idx]
                L_boot = L_group[idx]
                
                # Reference: never smoke
                res_never = model.simulate_gformula(G_boot, L_boot, 'never_smoke', n_time, n_mc // 2)
                risk_never = res_never['mean_cumulative_risk']
                
                # Quit at year qt
                res_quit = model.simulate_gformula(
                    G_boot, L_boot, f'quit_at_t{qt}', n_time, n_mc // 2, quit_time=qt
                )
                risk_quit = res_quit['mean_cumulative_risk']
                
                rr = risk_quit / (risk_never + 1e-10)
                rr_bootstrap.append(rr)
            
            rr_array = np.array(rr_bootstrap)
            
            results.append({
                'quit_year': qt,
                'group': group_name,
                'rr_mean': np.mean(rr_array),
                'rr_lower': np.percentile(rr_array, 2.5),
                'rr_upper': np.percentile(rr_array, 97.5),
                'rr_std': np.std(rr_array),
            })
    
    return pd.DataFrame(results)


# =============================================================================
# Visualization with 95% CI Shading
# =============================================================================

def smooth_spline(x: np.ndarray, y: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic spline 보간"""
    if len(x) < 4:
        return x, y
    
    try:
        # 정렬
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Smoothing
        y_smooth = gaussian_filter1d(y_sorted, sigma=0.8)
        
        # Spline
        x_fine = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
        spline = make_interp_spline(x_sorted, y_smooth, k=3)
        y_fine = spline(x_fine)
        
        return x_fine, y_fine
    except:
        return x, y


def plot_curve_a_prs_effect_with_ci(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 7),
) -> plt.Figure:
    """
    Curve A: PRS vs Risk Ratio with 95% CI shading
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Data
    x = df['prs'].values
    y_mean = df['rr_mean'].values
    y_lower = df['rr_lower'].values
    y_upper = df['rr_upper'].values
    
    # Sort by PRS
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y_mean = y_mean[sort_idx]
    y_lower = y_lower[sort_idx]
    y_upper = y_upper[sort_idx]
    
    # Spline interpolation for smooth curves
    x_fine, y_mean_fine = smooth_spline(x, y_mean)
    _, y_lower_fine = smooth_spline(x, y_lower)
    _, y_upper_fine = smooth_spline(x, y_upper)
    
    # Plot 95% CI band
    ax.fill_between(
        x_fine, y_lower_fine, y_upper_fine,
        alpha=0.25, color='royalblue', label='95% CI'
    )
    
    # Plot mean line
    ax.plot(x_fine, y_mean_fine, 'b-', linewidth=2.5, label='Mean Risk Ratio')
    
    # Plot raw points
    ax.scatter(x, y_mean, c='royalblue', s=50, alpha=0.7, edgecolors='white', zorder=5)
    
    # Reference line
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='RR = 1')
    
    # Labels
    ax.set_xlabel('Polygenic Risk Score (PRS)', fontsize=13)
    ax.set_ylabel('Risk Ratio (Always Smoke / Never Smoke)', fontsize=13)
    ax.set_title('Effect Modification: Smoking Effect by Genetic Risk\n(with 95% Bootstrap CI)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Annotation
    ax.annotate(
        'Higher genetic risk\n→ Greater harm from smoking',
        xy=(x.max() * 0.6, y_mean.max() * 0.85),
        fontsize=11, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_curve_b_quit_timing_with_ci(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 7),
) -> plt.Figure:
    """
    Curve B: Quit Year vs Risk Ratio with 95% CI shading
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'All': 'black', 'High_PRS': 'crimson', 'Low_PRS': 'forestgreen'}
    labels = {'All': 'All Subjects', 'High_PRS': 'High PRS (Top 20%)', 'Low_PRS': 'Low PRS (Bottom 80%)'}
    alphas = {'All': 0.20, 'High_PRS': 0.15, 'Low_PRS': 0.15}
    
    for group in df['group'].unique():
        df_g = df[df['group'] == group].sort_values('quit_year')
        
        x = df_g['quit_year'].values
        y_mean = df_g['rr_mean'].values
        y_lower = df_g['rr_lower'].values
        y_upper = df_g['rr_upper'].values
        
        color = colors.get(group, 'blue')
        alpha = alphas.get(group, 0.2)
        label = labels.get(group, group)
        
        # Spline interpolation
        if len(x) >= 4:
            x_fine, y_mean_fine = smooth_spline(x, y_mean, n_points=100)
            _, y_lower_fine = smooth_spline(x, y_lower, n_points=100)
            _, y_upper_fine = smooth_spline(x, y_upper, n_points=100)
            
            # 95% CI band
            ax.fill_between(
                x_fine, y_lower_fine, y_upper_fine,
                alpha=alpha, color=color
            )
            
            # Mean line
            ax.plot(x_fine, y_mean_fine, '-', color=color, linewidth=2.5, label=label)
        else:
            ax.fill_between(x, y_lower, y_upper, alpha=alpha, color=color)
            ax.plot(x, y_mean, '-', color=color, linewidth=2.5, label=label)
        
        # Raw points
        ax.scatter(x, y_mean, c=color, s=60, alpha=0.8, edgecolors='white', zorder=5)
    
    # Reference line
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='RR = 1 (No Excess Risk)')
    
    # Labels
    ax.set_xlabel('Year of Smoking Cessation', fontsize=13)
    ax.set_ylabel('Risk Ratio (vs. Never Smoked)', fontsize=13)
    ax.set_title('Urgency of Cessation: Effect of Delay on CVD Risk\n(with 95% Bootstrap CI)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Dynamic x-axis based on data
    max_year = int(df['quit_year'].max())
    xticks = list(range(0, max_year + 1, 4))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'Year {i}' for i in xticks])
    ax.set_xlim(0, max_year)
    
    # Annotation
    ax.annotate(
        'Earlier cessation → Greater benefit',
        xy=(max_year * 0.15, df[df['group'] == 'All']['rr_mean'].min() * 1.05),
        fontsize=11, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_combined_figure_with_ci(
    df_prs: pd.DataFrame,
    df_quit: pd.DataFrame,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 7),
) -> plt.Figure:
    """Combined figure with both analyses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ===== Panel A: PRS Effect =====
    x = df_prs['prs'].values
    y_mean = df_prs['rr_mean'].values
    y_lower = df_prs['rr_lower'].values
    y_upper = df_prs['rr_upper'].values
    
    sort_idx = np.argsort(x)
    x, y_mean, y_lower, y_upper = x[sort_idx], y_mean[sort_idx], y_lower[sort_idx], y_upper[sort_idx]
    
    x_fine, y_mean_fine = smooth_spline(x, y_mean)
    _, y_lower_fine = smooth_spline(x, y_lower)
    _, y_upper_fine = smooth_spline(x, y_upper)
    
    ax1.fill_between(x_fine, y_lower_fine, y_upper_fine, alpha=0.25, color='royalblue')
    ax1.plot(x_fine, y_mean_fine, 'b-', linewidth=2.5)
    ax1.scatter(x, y_mean, c='royalblue', s=40, alpha=0.7, edgecolors='white')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Polygenic Risk Score (PRS)', fontsize=12)
    ax1.set_ylabel('Risk Ratio (Smoke vs. Never)', fontsize=12)
    ax1.set_title('A. Effect Modification by Genetic Risk', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # ===== Panel B: Quit Timing =====
    colors = {'All': 'black', 'High_PRS': 'crimson', 'Low_PRS': 'forestgreen'}
    labels = {'All': 'All', 'High_PRS': 'High PRS', 'Low_PRS': 'Low PRS'}
    
    for group in df_quit['group'].unique():
        df_g = df_quit[df_quit['group'] == group].sort_values('quit_year')
        
        x_q = df_g['quit_year'].values
        y_q = df_g['rr_mean'].values
        y_q_lower = df_g['rr_lower'].values
        y_q_upper = df_g['rr_upper'].values
        
        color = colors.get(group, 'blue')
        label = labels.get(group, group)
        
        if len(x_q) >= 4:
            x_fine, y_fine = smooth_spline(x_q, y_q, 100)
            _, y_lower_fine = smooth_spline(x_q, y_q_lower, 100)
            _, y_upper_fine = smooth_spline(x_q, y_q_upper, 100)
            
            ax2.fill_between(x_fine, y_lower_fine, y_upper_fine, alpha=0.15, color=color)
            ax2.plot(x_fine, y_fine, '-', color=color, linewidth=2.5, label=label)
        
        ax2.scatter(x_q, y_q, c=color, s=50, alpha=0.7, edgecolors='white')
    
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Year of Cessation', fontsize=12)
    ax2.set_ylabel('Risk Ratio (vs. Never Smoked)', fontsize=12)
    ax2.set_title('B. Urgency of Smoking Cessation', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Gene-Environment Interaction in CVD Risk (95% CI Bands)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_advanced_analysis(
    n_samples: int = 20000,
    n_time: int = 20,  # 20년 시뮬레이션
    n_bootstrap: int = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """전체 심화 분석 실행"""
    n_bootstrap = n_bootstrap or MC_PARAMS.get('n_bootstrap', 200)
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS: Spline Curves with 95% CI (v3.2)")
    print("="*70)
    print(f"N samples: {n_samples:,}, Bootstrap iterations: {n_bootstrap}")
    
    # 데이터 생성 및 모델 학습
    set_seed(SEED)
    data = generate_synthetic_data(n_samples=n_samples, n_time=n_time, seed=SEED)
    
    if verbose:
        validate_dgp(data)
    
    print("\nFitting model...")
    model = HiddenMarkovGFormula(
        n_covariates=N_COVARIATES, 
        fit_interaction=True, 
        fit_pack_years=True,
        fit_time_effects=True
    )
    model.fit(data.G, data.S, data.C, data.Y, L=data.L, n_epochs=150, verbose=False)
    
    if verbose:
        print("Model parameters:")
        params = model.get_parameters()
        for k in ['beta_GS', 'beta_time', 'alpha_time']:
            if k in params:
                print(f"  {k}: {params[k]:.4f}")
    
    # Analysis A: PRS Effect with Bootstrap CI
    print("\n[Analysis A] PRS Effect Modification with Bootstrap CI...")
    df_prs = analyze_prs_effect_with_bootstrap(
        model, data.G, data.L, 
        n_prs_points=12, 
        n_bootstrap=min(n_bootstrap, 80),
        n_time=n_time,
        verbose=verbose
    )
    
    # Analysis B: Quit Timing with Bootstrap CI
    print("\n[Analysis B] Quit Timing Effect with Bootstrap CI...")
    # 2년 간격으로 0부터 n_time까지
    target_quit_years = list(range(0, n_time + 1, 2))
    df_quit = analyze_quit_timing_with_bootstrap(
        model, data.G, data.L,
        quit_years=target_quit_years,
        n_bootstrap=min(n_bootstrap, 50),
        n_time=n_time,
        prs_groups=True,
        verbose=verbose
    )
    
    # Visualization
    print("\nGenerating figures with 95% CI bands...")
    
    fig_a = plot_curve_a_prs_effect_with_ci(
        df_prs,
        save_path=os.path.join(OUTPUT_DIR, 'curve_a_prs_effect_95ci.png') if save_results else None
    )
    
    fig_b = plot_curve_b_quit_timing_with_ci(
        df_quit,
        save_path=os.path.join(OUTPUT_DIR, 'curve_b_quit_timing_95ci.png') if save_results else None
    )
    
    fig_combined = plot_combined_figure_with_ci(
        df_prs, df_quit,
        save_path=os.path.join(OUTPUT_DIR, 'figure_combined_95ci.png') if save_results else None
    )
    
    # Save data
    if save_results:
        df_prs.to_csv(os.path.join(OUTPUT_DIR, 'analysis_prs_effect_ci.csv'), index=False)
        df_quit.to_csv(os.path.join(OUTPUT_DIR, 'analysis_quit_timing_ci.csv'), index=False)
        print(f"\nData saved to {OUTPUT_DIR}/")
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS COMPLETED")
    print("="*70)
    
    plt.close('all')
    
    return {
        'prs_effect': df_prs,
        'quit_timing': df_quit,
        'model_params': model.get_parameters(),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=20000)
    parser.add_argument('--n-bootstrap', type=int, default=100)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    if args.quick:
        args.n_samples = 5000
        args.n_bootstrap = 30
    
    run_advanced_analysis(
        n_samples=args.n_samples,
        n_bootstrap=args.n_bootstrap,
    )