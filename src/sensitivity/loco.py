"""Leave-one-covariate-out + grouped exclusion sensitivity wrapper."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

from ..benchmarks.base import BenchmarkMethod, DoseResponseResult
from ..data.ards import ARDSConfig, load_ards_cohort, TV_COLS


@dataclass
class GroupExclusion:
    """Named domain group of covariates to exclude jointly."""
    label: str
    tv_cols: tuple[str, ...] = ()
    static_cols: tuple[str, ...] = ()


@dataclass
class LOCOResult:
    """Per-feature dose-response under single-covariate or grouped exclusion."""
    method_name: str
    by_excluded_tv: dict[str, DoseResponseResult] = field(default_factory=dict)
    by_excluded_static: dict[str, DoseResponseResult] = field(default_factory=dict)
    by_excluded_group: dict[str, DoseResponseResult] = field(default_factory=dict)


def run_loco(
    method_factory,
    base_cfg: ARDSConfig,
    target_bins: Sequence[int],
    n_bootstrap: int = 50,
    seed: int = 0,
    refit: bool = True,
    tv_cols: Sequence[str] = tuple(TV_COLS),
    static_cols: Sequence[str] = (),
    groups: Sequence[GroupExclusion] = (),
) -> LOCOResult:
    """Refit a fresh method per (single covariate or named group) exclusion.

    `method_factory` must be a zero-argument callable returning a fresh
    BenchmarkMethod instance. `refit` is forwarded to dose_response.
    `groups` defines clinical-scenario covariate sets to exclude jointly.
    """
    sample = method_factory()
    method_name = getattr(sample, "method_name", "method")
    result = LOCOResult(method_name=method_name)

    for c in tv_cols:
        cfg = ARDSConfig(**{**base_cfg.__dict__, "exclude_tv_cols": (c,)})
        cohort = load_ards_cohort(cfg)
        m: BenchmarkMethod = method_factory()
        result.by_excluded_tv[c] = m.dose_response(
            cohort, target_bins=target_bins,
            n_bootstrap=n_bootstrap, seed=seed, refit=refit,
        )

    for c in static_cols:
        cfg = ARDSConfig(**{**base_cfg.__dict__, "exclude_static_cols": (c,)})
        cohort = load_ards_cohort(cfg)
        m = method_factory()
        result.by_excluded_static[c] = m.dose_response(
            cohort, target_bins=target_bins,
            n_bootstrap=n_bootstrap, seed=seed, refit=refit,
        )

    for grp in groups:
        cfg = ARDSConfig(**{
            **base_cfg.__dict__,
            "exclude_tv_cols": tuple(grp.tv_cols),
            "exclude_static_cols": tuple(grp.static_cols),
        })
        cohort = load_ards_cohort(cfg)
        m = method_factory()
        result.by_excluded_group[grp.label] = m.dose_response(
            cohort, target_bins=target_bins,
            n_bootstrap=n_bootstrap, seed=seed, refit=refit,
        )

    return result
