"""
utils/visualization.py - Visualization Utilities

시뮬레이션 결과 시각화:
- Parameter recovery plots
- Method comparison plots
- g-formula risk trajectory plots
- Bias-variance trade-off plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FIGURE_SIZE_SINGLE = (8, 6)
FIGURE_SIZE_DOUBLE = (12, 5)
FIGURE_SIZE_MULTI = (14, 10)


def plot_parameter_recovery(
    estimates_by_scenario: Dict[str, np.ndarray],
    true_values: Dict[str, float],
    param_name: str = 'beta_GS',
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_SINGLE,
) -> plt.Figure:
    """
    Effect size별 parameter recovery 시각화
    
    Args:
        estimates_by_scenario: {scenario_name: (n_sim,) estimates}
        true_values: {scenario_name: true_value}
        param_name: Parameter name for axis label
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for boxplot
    data = []
    for scenario, estimates in estimates_by_scenario.items():
        for est in estimates:
            data.append({'Scenario': scenario, 'Estimate': est})
    df = pd.DataFrame(data)
    
    # Boxplot
    box = sns.boxplot(
        x='Scenario', y='Estimate', data=df, ax=ax,
        palette='Set2', width=0.6
    )
    
    # Add true value lines
    scenarios = list(estimates_by_scenario.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(scenarios)))
    
    for idx, scenario in enumerate(scenarios):
        true_val = true_values.get(scenario, 0)
        ax.axhline(
            y=true_val, 
            xmin=(idx + 0.1) / len(scenarios), 
            xmax=(idx + 0.9) / len(scenarios),
            color='red', 
            linestyle='--', 
            linewidth=2,
            alpha=0.7
        )
        # Annotate true value
        ax.annotate(
            f'True={true_val:.2f}',
            xy=(idx, true_val),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            color='red'
        )
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel(f'Estimated {param_name}', fontsize=12)
    ax.set_title(title or f'Parameter Recovery: {param_name}', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sample_size_robustness(
    estimates_by_n: Dict[int, np.ndarray],
    true_value: float,
    param_name: str = 'beta_GS',
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_SINGLE,
) -> plt.Figure:
    """
    Sample size에 따른 추정 안정성 시각화 (Violin plot)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    data = []
    for n, estimates in estimates_by_n.items():
        for est in estimates:
            data.append({'Sample Size': str(n), 'Estimate': est, 'N': n})
    df = pd.DataFrame(data)
    
    # Sort by sample size
    df['N'] = df['N'].astype(int)
    df = df.sort_values('N')
    df['Sample Size'] = df['Sample Size'].astype(str)
    
    # Violin plot
    sns.violinplot(
        x='Sample Size', y='Estimate', data=df, ax=ax,
        palette='Blues', inner='quartile',
        order=[str(n) for n in sorted(estimates_by_n.keys())]
    )
    
    # True value line
    ax.axhline(y=true_value, color='red', linestyle='--', linewidth=2, label=f'True Value ({true_value})')
    
    ax.set_xlabel('Sample Size (N)', fontsize=12)
    ax.set_ylabel(f'Estimated {param_name}', fontsize=12)
    ax.set_title(title or f'Robustness Analysis: Effect of Sample Size on {param_name}', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_method_comparison(
    results_by_method: Dict[str, Dict[str, float]],
    metrics: List[str] = ['bias', 'rmse', 'coverage'],
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_DOUBLE,
) -> plt.Figure:
    """
    방법론 비교 bar plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    methods = list(results_by_method.keys())
    x = np.arange(len(methods))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = [results_by_method[m].get(metric, 0) for m in methods]
        
        bars = ax.bar(x, values, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()}')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=9
            )
        
        # Reference lines
        if metric == 'coverage':
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Nominal (0.95)')
            ax.legend()
        elif metric == 'bias':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.suptitle(title or 'Method Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gformula_trajectories(
    trajectories: Dict[str, List[float]],
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_SINGLE,
) -> plt.Figure:
    """
    g-formula 시뮬레이션 결과: 개입별 위험도 궤적
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    markers = ['o-', 's--', '^:', 'd-.']
    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
    
    for idx, (intervention, trajectory) in enumerate(trajectories.items()):
        time_points = range(len(trajectory))
        ax.plot(
            time_points, trajectory,
            markers[idx % len(markers)],
            color=colors[idx],
            linewidth=2,
            markersize=6,
            label=intervention
        )
    
    ax.set_xlabel('Time (Years)', fontsize=12)
    ax.set_ylabel('CVD Risk (Probability)', fontsize=12)
    ax.set_title(title or 'g-formula Simulation: Intervention Effects', fontsize=14)
    ax.legend(title='Intervention')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_bias_variance_tradeoff(
    results_by_method: Dict[str, Dict[str, np.ndarray]],
    true_value: float,
    param_name: str = 'beta_GS',
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_SINGLE,
) -> plt.Figure:
    """
    Bias vs Variance trade-off scatter plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_by_method)))
    
    for idx, (method, params_dict) in enumerate(results_by_method.items()):
        if param_name not in params_dict:
            continue
            
        estimates = params_dict[param_name]
        bias = np.abs(np.mean(estimates) - true_value)
        variance = np.var(estimates)
        
        ax.scatter(
            bias, variance,
            s=150,
            c=[colors[idx]],
            label=method,
            alpha=0.8,
            edgecolors='black'
        )
    
    ax.set_xlabel('|Bias|', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title(title or f'Bias-Variance Trade-off: {param_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_convergence(
    losses_by_method: Dict[str, List[float]],
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_SINGLE,
) -> plt.Figure:
    """
    Training convergence plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for method, losses in losses_by_method.items():
        ax.plot(losses, label=method, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title or 'Training Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale if values vary a lot
    all_losses = [l for losses in losses_by_method.values() for l in losses if l > 0]
    if len(all_losses) > 0 and max(all_losses) / min(all_losses) > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_figure(
    exp1_results: Dict,
    exp2_results: Dict,
    exp3_results: Dict,
    gformula_results: Dict,
    save_path: str = None,
    figsize: Tuple[int, int] = FIGURE_SIZE_MULTI,
) -> plt.Figure:
    """
    전체 실험 결과를 하나의 figure로 요약
    """
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Parameter Recovery
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('A. Parameter Recovery by Effect Size')
    ax1.text(0.5, 0.5, 'Parameter Recovery Plot', ha='center', va='center', transform=ax1.transAxes)
    
    # Panel B: Sample Size Robustness
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('B. Robustness to Sample Size')
    ax2.text(0.5, 0.5, 'Sample Size Plot', ha='center', va='center', transform=ax2.transAxes)
    
    # Panel C: Method Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('C. Method Comparison')
    ax3.text(0.5, 0.5, 'Method Comparison Plot', ha='center', va='center', transform=ax3.transAxes)
    
    # Panel D: g-formula Simulation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('D. g-formula Intervention Effects')
    ax4.text(0.5, 0.5, 'g-formula Plot', ha='center', va='center', transform=ax4.transAxes)
    
    plt.suptitle('Simulation Study Summary', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Testing Visualization Module...")
    
    # Generate dummy data
    np.random.seed(42)
    
    # Test 1: Parameter recovery
    estimates = {
        'Null': np.random.normal(0.0, 0.05, 100),
        'Weak': np.random.normal(0.15, 0.06, 100),
        'Moderate': np.random.normal(0.40, 0.07, 100),
        'Strong': np.random.normal(0.60, 0.08, 100),
    }
    true_vals = {'Null': 0.0, 'Weak': 0.15, 'Moderate': 0.40, 'Strong': 0.60}
    
    fig1 = plot_parameter_recovery(estimates, true_vals, save_path='test_param_recovery.png')
    print("Saved: test_param_recovery.png")
    
    # Test 2: Sample size
    estimates_n = {
        1000: np.random.normal(0.4, 0.15, 100),
        5000: np.random.normal(0.4, 0.08, 100),
        10000: np.random.normal(0.4, 0.05, 100),
        20000: np.random.normal(0.4, 0.03, 100),
    }
    
    fig2 = plot_sample_size_robustness(estimates_n, 0.4, save_path='test_sample_size.png')
    print("Saved: test_sample_size.png")
    
    # Test 3: Method comparison
    methods_results = {
        'HMM-gFormula': {'bias': 0.01, 'rmse': 0.05, 'coverage': 0.94},
        'Naive Logistic': {'bias': 0.15, 'rmse': 0.18, 'coverage': 0.72},
        'MSM-IPTW': {'bias': 0.05, 'rmse': 0.10, 'coverage': 0.88},
    }
    
    fig3 = plot_method_comparison(methods_results, save_path='test_method_comp.png')
    print("Saved: test_method_comp.png")
    
    # Test 4: g-formula
    trajectories = {
        'Natural': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
        'Never Smoke': [0.008, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055],
        'Always Smoke': [0.015, 0.035, 0.055, 0.075, 0.095, 0.115, 0.135, 0.155, 0.175, 0.195],
    }
    
    fig4 = plot_gformula_trajectories(trajectories, save_path='test_gformula.png')
    print("Saved: test_gformula.png")
    
    plt.close('all')
    print("\nAll tests completed!")