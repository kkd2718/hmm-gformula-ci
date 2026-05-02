"""LOCO sensitivity for Standard and Xu (parallel to VEM LOCO in Table 3).

Runs single-covariate LOCO + grouped exclusion for Standard parametric g-formula
and Xu 2024 GLMM, sequentially. Output mirrors VEM LOCO format from
table3_sensitivity.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, TV_COLS, STATIC_COLS
from src.sensitivity.loco import run_loco
from src.benchmarks import StandardGFormula, XuGLMM
from scripts.table3_sensitivity import CLINICAL_GROUPS


def _save_loco(loco_result, out_dir: Path, method_name: str, target_bins: list[int],
               centers: np.ndarray, full_risk_mean: np.ndarray) -> None:
    """Write per-bin LOCO range table for one method."""
    K = len(target_bins)

    def _stack(d):
        labels = list(d.keys())
        if not labels:
            return np.zeros((0, K)), []
        return np.stack([d[k].risk_mean for k in labels]), labels

    tv_means, tv_labels = _stack(loco_result.by_excluded_tv)
    static_means, static_labels = _stack(loco_result.by_excluded_static)
    group_means, group_labels = _stack(loco_result.by_excluded_group)

    flat = {}
    for col, res in loco_result.by_excluded_tv.items():
        flat[f"loco_tv__{col}__risk_mean"] = res.risk_mean
        flat[f"loco_tv__{col}__risk_ci_low"] = res.risk_ci_low
        flat[f"loco_tv__{col}__risk_ci_high"] = res.risk_ci_high
    for col, res in loco_result.by_excluded_static.items():
        flat[f"loco_static__{col}__risk_mean"] = res.risk_mean
        flat[f"loco_static__{col}__risk_ci_low"] = res.risk_ci_low
        flat[f"loco_static__{col}__risk_ci_high"] = res.risk_ci_high
    for label, res in loco_result.by_excluded_group.items():
        safe = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        flat[f"group__{safe}__risk_mean"] = res.risk_mean
        flat[f"group__{safe}__risk_ci_low"] = res.risk_ci_low
        flat[f"group__{safe}__risk_ci_high"] = res.risk_ci_high
    np.savez(out_dir / f"loco_{method_name}.npz", **flat)

    md = [
        f"# Appendix G — LOCO sensitivity ({method_name})",
        "",
        f"_Per-bin point-estimate range across LOCO refits vs full-covariate primary._",
        "",
        "| MP bin | Center (J/min) | Full primary (%) | TV LOCO range (%) | Static LOCO range (%) | Grouped exclusion range (%) |",
        "|---|---|---|---|---|---|",
    ]
    for k in target_bins:
        c = f"{centers[k]:.1f}" if np.isfinite(centers[k]) else "-"
        full = f"{full_risk_mean[k] * 100:.1f}"
        tv_rng = (
            f"[{tv_means[:, k].min() * 100:.1f}, {tv_means[:, k].max() * 100:.1f}]"
            if len(tv_labels) else "—"
        )
        st_rng = (
            f"[{static_means[:, k].min() * 100:.1f}, {static_means[:, k].max() * 100:.1f}]"
            if len(static_labels) else "—"
        )
        gr_rng = (
            f"[{group_means[:, k].min() * 100:.1f}, {group_means[:, k].max() * 100:.1f}]"
            if len(group_labels) else "—"
        )
        md.append(f"| {k} | {c} | {full} | {tv_rng} | {st_rng} | {gr_rng} |")

    md.extend([
        "",
        "### Per-exclusion detail at reference (bin 16) and high (bin 18) bins",
        "",
        "| Exclusion | Type | Risk @ ref (%) | Risk @ high (%) |",
        "|---|---|---|---|",
        f"| (full primary) | — | {full_risk_mean[16] * 100:.1f} | {full_risk_mean[18] * 100:.1f} |",
    ])
    for label, res in loco_result.by_excluded_tv.items():
        md.append(
            f"| {label} | TV | {res.risk_mean[16] * 100:.1f} | {res.risk_mean[18] * 100:.1f} |"
        )
    for label, res in loco_result.by_excluded_static.items():
        md.append(
            f"| {label} | Static | {res.risk_mean[16] * 100:.1f} | {res.risk_mean[18] * 100:.1f} |"
        )
    for label, res in loco_result.by_excluded_group.items():
        md.append(
            f"| {label} | Group | {res.risk_mean[16] * 100:.1f} | {res.risk_mean[18] * 100:.1f} |"
        )

    md_path = out_dir / f"loco_{method_name}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--npz", required=True, type=Path,
                        help="Path to table2_risks.npz to reuse bin centers + ref bin.")
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--n-bootstrap", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["standard", "xu"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.npz)
    centers = z["standard_gformula__bin_centers_J_min"]
    target_bins = list(range(args.n_bins))
    base_cfg = ARDSConfig(csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t)

    if "standard" in args.methods:
        print("\n=== Standard parametric g-formula LOCO ===")
        loco_std = run_loco(
            method_factory=lambda: StandardGFormula(),
            base_cfg=base_cfg,
            target_bins=target_bins,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            refit=False,
            tv_cols=tuple(TV_COLS),
            static_cols=tuple(STATIC_COLS),
            groups=CLINICAL_GROUPS,
        )
        _save_loco(loco_std, args.out_dir, "standard", target_bins, centers,
                   z["standard_gformula__risk_mean"])

    if "xu" in args.methods:
        print("\n=== Xu 2024 GLMM LOCO ===")
        loco_xu = run_loco(
            method_factory=lambda: XuGLMM(n_b_draws=50),
            base_cfg=base_cfg,
            target_bins=target_bins,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            refit=False,
            tv_cols=tuple(TV_COLS),
            static_cols=tuple(STATIC_COLS),
            groups=CLINICAL_GROUPS,
        )
        _save_loco(loco_xu, args.out_dir, "xu", target_bins, centers,
                   z["xu_glmm__risk_mean"])


if __name__ == "__main__":
    main()
