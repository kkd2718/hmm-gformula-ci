"""Benchmark methods for the 4-way ARDS-MP causal comparison.

Methods (post-Bayesian-pivot):
  1. StandardGFormula             — frequentist NICE g-formula, no RE (baseline)
  2. XuGLMMBayesian               — Bayesian GLMM, scalar RE, MSM observed-L (Xu 2024)
  3. FRENICEBayesianBenchmark(K=1)— Bayesian, scalar RE, NICE forward-L (= Xu's RE in NICE framework)
  4. FRENICEBayesianBenchmark(K=5)— Bayesian, functional RE via spline, NICE (proposed)

Frequentist Laplace variants (xu_glmm.XuGLMM, spline_glmm_nice.SplineGLMMNICEBenchmark)
retained for sensitivity / LOCO speed but not used as primary baseline.
"""
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from .standard_gformula import StandardGFormula
from .xu_glmm import XuGLMM                                  # legacy frequentist
from .xu_glmm_bayesian import XuGLMMBayesian, XuBayesianConfig
from .proposed import VEMSSMBenchmark, VEMConfig             # legacy SSM
from .spline_glmm_nice import (
    SplineGLMMNICEBenchmark, SplineGLMMNICEConfig,
)                                                            # legacy frequentist
from .fre_nice_bayesian import (
    FRENICEBayesianBenchmark, FRENICEBayesianConfig,
)

__all__ = [
    "BenchmarkMethod", "DoseResponseResult", "bin_centers_J_min",
    "StandardGFormula",
    "XuGLMM",                          # legacy
    "XuGLMMBayesian", "XuBayesianConfig",
    "VEMSSMBenchmark", "VEMConfig",    # legacy
    "SplineGLMMNICEBenchmark", "SplineGLMMNICEConfig",   # legacy
    "FRENICEBayesianBenchmark", "FRENICEBayesianConfig",
]
