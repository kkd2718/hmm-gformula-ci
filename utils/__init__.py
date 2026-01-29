"""
utils/__init__.py - Utilities Package
"""

from .metrics import (
    compute_bias,
    compute_variance,
    compute_mse,
    compute_rmse,
    compute_coverage,
    compute_power,
    evaluate_parameter,
    evaluate_all_parameters,
    compare_methods,
    create_summary_table,
    ParameterMetrics,
)

from .visualization import (
    plot_parameter_recovery,
    plot_sample_size_robustness,
    plot_method_comparison,
    plot_gformula_trajectories,
    plot_bias_variance_tradeoff,
    plot_convergence,
)

__all__ = [
    # Metrics
    'compute_bias',
    'compute_variance',
    'compute_mse',
    'compute_rmse',
    'compute_coverage',
    'compute_power',
    'evaluate_parameter',
    'evaluate_all_parameters',
    'compare_methods',
    'create_summary_table',
    'ParameterMetrics',
    # Visualization
    'plot_parameter_recovery',
    'plot_sample_size_robustness',
    'plot_method_comparison',
    'plot_gformula_trajectories',
    'plot_bias_variance_tradeoff',
    'plot_convergence',
]