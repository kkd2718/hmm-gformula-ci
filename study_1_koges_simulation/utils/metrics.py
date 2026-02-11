"""
utils/metrics.py - Statistical Evaluation Metrics

시뮬레이션 스터디에 필요한 통계적 지표:
- Bias
- MSE (Mean Squared Error)
- Coverage Probability
- Power / Type I Error
- RMSE
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class ParameterMetrics:
    """단일 파라미터에 대한 평가 지표"""
    true_value: float
    estimates: np.ndarray
    bias: float
    variance: float
    mse: float
    rmse: float
    coverage: float  # 95% CI coverage
    power: Optional[float] = None  # For testing H0: param = 0
    
    def __repr__(self):
        return (
            f"ParameterMetrics(\n"
            f"  true={self.true_value:.4f},\n"
            f"  mean_est={self.estimates.mean():.4f},\n"
            f"  bias={self.bias:.4f},\n"
            f"  rmse={self.rmse:.4f},\n"
            f"  coverage={self.coverage:.3f}"
            f")"
        )


def compute_bias(estimates: np.ndarray, true_value: float) -> float:
    """
    Bias = E[θ̂] - θ
    """
    return np.mean(estimates) - true_value


def compute_variance(estimates: np.ndarray) -> float:
    """
    Variance = Var(θ̂)
    """
    return np.var(estimates, ddof=1)


def compute_mse(estimates: np.ndarray, true_value: float) -> float:
    """
    MSE = E[(θ̂ - θ)²] = Var(θ̂) + Bias²
    """
    return np.mean((estimates - true_value) ** 2)


def compute_rmse(estimates: np.ndarray, true_value: float) -> float:
    """
    RMSE = √MSE
    """
    return np.sqrt(compute_mse(estimates, true_value))


def compute_coverage(
    estimates: np.ndarray,
    std_errors: np.ndarray,
    true_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Coverage probability: proportion of CIs that contain true value
    
    Args:
        estimates: Point estimates from simulations
        std_errors: Standard errors from simulations
        true_value: True parameter value
        confidence_level: Confidence level (default 0.95)
        
    Returns:
        Coverage probability
    """
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    lower = estimates - z * std_errors
    upper = estimates + z * std_errors
    
    covered = (lower <= true_value) & (true_value <= upper)
    
    return np.mean(covered)


def compute_coverage_bootstrap(
    estimates: np.ndarray,
    true_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Coverage using percentile bootstrap CI
    (Standard error 없이 estimate 분포로부터 직접 계산)
    """
    alpha = 1 - confidence_level
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    
    # True value가 CI 안에 있으면 coverage = 1
    covered = (lower <= true_value) & (true_value <= upper)
    
    return float(covered)


def compute_power(
    estimates: np.ndarray,
    std_errors: np.ndarray,
    null_value: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """
    Power: proportion of simulations that reject H0: θ = null_value
    
    For null_value=0, this is power when true effect exists,
    or Type I error when true effect is 0.
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    z_stats = np.abs(estimates - null_value) / (std_errors + 1e-10)
    rejected = z_stats > z_crit
    
    return np.mean(rejected)


def compute_power_from_ci(
    estimates: np.ndarray,
    null_value: float = 0.0,
    confidence_level: float = 0.95,
) -> float:
    """
    Power using bootstrap CI (does CI exclude null_value?)
    """
    alpha = 1 - confidence_level
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    
    # null_value가 CI 밖에 있으면 reject
    rejected = (upper < null_value) | (lower > null_value)
    
    return float(rejected)


def evaluate_parameter(
    estimates: np.ndarray,
    true_value: float,
    std_errors: Optional[np.ndarray] = None,
    param_name: str = 'parameter',
) -> ParameterMetrics:
    """
    단일 파라미터에 대한 모든 평가 지표 계산
    
    Args:
        estimates: (n_simulations,) array of estimates
        true_value: True parameter value
        std_errors: (n_simulations,) array of standard errors (optional)
        param_name: Parameter name for reporting
        
    Returns:
        ParameterMetrics dataclass
    """
    estimates = np.asarray(estimates)
    
    bias = compute_bias(estimates, true_value)
    variance = compute_variance(estimates)
    mse = compute_mse(estimates, true_value)
    rmse = compute_rmse(estimates, true_value)
    
    # Coverage
    if std_errors is not None:
        std_errors = np.asarray(std_errors)
        coverage = compute_coverage(estimates, std_errors, true_value)
        power = compute_power(estimates, std_errors, null_value=0.0)
    else:
        coverage = compute_coverage_bootstrap(estimates, true_value)
        power = compute_power_from_ci(estimates, null_value=0.0)
    
    return ParameterMetrics(
        true_value=true_value,
        estimates=estimates,
        bias=bias,
        variance=variance,
        mse=mse,
        rmse=rmse,
        coverage=coverage,
        power=power,
    )


def evaluate_all_parameters(
    all_estimates: Dict[str, np.ndarray],
    true_params: Dict[str, float],
    all_std_errors: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, ParameterMetrics]:
    """
    모든 파라미터에 대한 평가 수행
    
    Args:
        all_estimates: {param_name: (n_sim,) array}
        true_params: {param_name: true_value}
        all_std_errors: {param_name: (n_sim,) array} (optional)
        
    Returns:
        {param_name: ParameterMetrics}
    """
    results = {}
    
    for param_name in all_estimates.keys():
        if param_name not in true_params:
            continue
            
        estimates = all_estimates[param_name]
        true_value = true_params[param_name]
        
        std_errors = None
        if all_std_errors is not None and param_name in all_std_errors:
            std_errors = all_std_errors[param_name]
        
        results[param_name] = evaluate_parameter(
            estimates, true_value, std_errors, param_name
        )
    
    return results


def create_summary_table(
    metrics_dict: Dict[str, ParameterMetrics],
    include_power: bool = True,
) -> str:
    """
    평가 결과를 표 형식의 문자열로 반환
    """
    header = f"{'Parameter':<15} {'True':>8} {'Mean Est':>10} {'Bias':>10} {'RMSE':>10} {'Coverage':>10}"
    if include_power:
        header += f" {'Power':>8}"
    
    lines = [header, "-" * len(header)]
    
    for param_name, metrics in metrics_dict.items():
        line = (
            f"{param_name:<15} "
            f"{metrics.true_value:>8.4f} "
            f"{metrics.estimates.mean():>10.4f} "
            f"{metrics.bias:>10.4f} "
            f"{metrics.rmse:>10.4f} "
            f"{metrics.coverage:>10.3f}"
        )
        if include_power and metrics.power is not None:
            line += f" {metrics.power:>8.3f}"
        lines.append(line)
    
    return "\n".join(lines)


# =============================================================================
# Model Comparison Metrics
# =============================================================================

def compare_methods(
    results_by_method: Dict[str, Dict[str, np.ndarray]],
    true_params: Dict[str, float],
    target_param: str = 'beta_GS',
) -> Dict[str, Dict[str, float]]:
    """
    여러 방법론의 성능 비교
    
    Args:
        results_by_method: {method_name: {param_name: estimates}}
        true_params: True parameter values
        target_param: Primary parameter for comparison
        
    Returns:
        {method_name: {metric: value}}
    """
    comparison = {}
    
    for method_name, estimates_dict in results_by_method.items():
        if target_param not in estimates_dict:
            continue
            
        estimates = np.asarray(estimates_dict[target_param])
        true_value = true_params.get(target_param, 0.0)
        
        metrics = evaluate_parameter(estimates, true_value)
        
        comparison[method_name] = {
            'bias': metrics.bias,
            'rmse': metrics.rmse,
            'coverage': metrics.coverage,
            'power': metrics.power,
        }
    
    return comparison


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Metrics Module...")
    
    # Simulate estimates
    np.random.seed(42)
    true_value = 0.4
    n_sim = 500
    
    # Good estimator (low bias, low variance)
    estimates_good = np.random.normal(true_value, 0.05, n_sim)
    
    # Biased estimator
    estimates_biased = np.random.normal(true_value + 0.1, 0.05, n_sim)
    
    # High variance estimator
    estimates_high_var = np.random.normal(true_value, 0.2, n_sim)
    
    print("\n1. Good Estimator:")
    metrics = evaluate_parameter(estimates_good, true_value)
    print(metrics)
    
    print("\n2. Biased Estimator:")
    metrics = evaluate_parameter(estimates_biased, true_value)
    print(metrics)
    
    print("\n3. High Variance Estimator:")
    metrics = evaluate_parameter(estimates_high_var, true_value)
    print(metrics)
    
    # Test summary table
    print("\n" + "="*60)
    print("Summary Table:")
    print("="*60)
    
    all_metrics = {
        'good': evaluate_parameter(estimates_good, true_value),
        'biased': evaluate_parameter(estimates_biased, true_value),
        'high_var': evaluate_parameter(estimates_high_var, true_value),
    }
    print(create_summary_table(all_metrics))