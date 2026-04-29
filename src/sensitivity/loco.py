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
    import gc
    try:
        import torch
    except ImportError:
        torch = None
    sample = method_factory()
    method_name = getattr(sample, "method_name", "method")
    result = LOCOResult(method_name=method_name)

    def _run_one(label: str, tag: str, exclude_tv, exclude_static):
        print(f"[LOCO] {tag} excluding {label!r} ...", flush=True)
        try:
            cfg = ARDSConfig(**{
                **base_cfg.__dict__,
                "exclude_tv_cols": tuple(exclude_tv),
                "exclude_static_cols": tuple(exclude_static),
            })
            cohort = load_ards_cohort(cfg)
            m: BenchmarkMethod = method_factory()
            res = m.dose_response(
                cohort, target_bins=target_bins,
                n_bootstrap=n_bootstrap, seed=seed, refit=refit,
            )
            del m, cohort
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[LOCO] {tag} excluding {label!r} done.", flush=True)
            return res
        except Exception as e:
            print(f"[LOCO][ERROR] {tag} {label!r}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
            return None

    for c in tv_cols:
        res = _run_one(c, "tv", [c], [])
        if res is not None:
            result.by_excluded_tv[c] = res

    for c in static_cols:
        res = _run_one(c, "static", [], [c])
        if res is not None:
            result.by_excluded_static[c] = res

    for grp in groups:
        res = _run_one(grp.label, "group", grp.tv_cols, grp.static_cols)
        if res is not None:
            result.by_excluded_group[grp.label] = res

    return result
