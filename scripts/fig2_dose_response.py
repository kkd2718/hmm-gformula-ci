"""Generate Figure 2: MP dose-response, two complementary visualizations.

Both styles read the same `<results_dir>/table2_risks.npz` and represent the
same per-bin point estimates and 95% CIs — they differ only in visual style:

(A) `fig2_dose_response_scatter.png` — scatter at bin centers with vertical 95%
    CI error bars; no connecting line. Most rigorous representation of the
    discrete binned counterfactual estimand (Costa 2021 AJRCCM / Serpa Neto
    2018 ICM convention).

(B) `fig2_dose_response_curves.png` — connected lines between bin centers with
    shaded 95% CI bands. Visually compelling but the connecting line is purely
    a guide-to-the-eye; the model does not estimate counterfactual risk
    between bin centers (Yarnell 2023 Crit Care Fig 2 style).

Both are produced so the writer can choose per manuscript.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

METHOD_LABELS = {
    "standard_gformula": ("Standard parametric g-formula", "#1F3A5F", "o"),
    "xu_glmm": ("Xu 2024 GLMM g-computation", "#A07A3F", "s"),
    "vem_ssm": ("VEM-SSM g-formula (proposed)", "#B31B1B", "D"),
}


def _gather(z, mp_min, mp_max):
    series: list[tuple[str, str, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for key, (label, color, marker) in METHOD_LABELS.items():
        if f"{key}__risk_mean" not in z.files:
            continue
        centers = z[f"{key}__bin_centers_J_min"]
        mean = z[f"{key}__risk_mean"]
        lo = z[f"{key}__risk_ci_low"]
        hi = z[f"{key}__risk_ci_high"]
        mask = np.isfinite(centers) & (centers >= mp_min) & (centers <= mp_max)
        x = centers[mask].astype(float)
        order = np.argsort(x)
        series.append((
            key, label, color, marker,
            x[order], (mean[mask] * 100)[order],
            (lo[mask] * 100)[order], (hi[mask] * 100)[order],
        ))
    return series


def _plot_scatter(series, ref_mp, log_x, out):
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    n = len(series)
    offset_step = 0.04
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * offset_step
    for i, (_, label, color, marker, x, y, lo, hi) in enumerate(series):
        x_off = x * (1.0 + offsets[i]) if log_x else x + offsets[i] * 5.0
        ax.errorbar(
            x_off, y, yerr=[y - lo, hi - y],
            fmt=marker, color=color, ecolor=color, lw=1.0,
            markersize=5, capsize=3, label=label, alpha=0.85,
        )
    ax.axvline(ref_mp, color="#7A7A7A", ls="--", lw=1.0, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(
        ref_mp, ymax * 0.97,
        f"  MP = {ref_mp:g} J/min\n  (Costa 2021)",
        color="#7A7A7A", fontsize=8, va="top",
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Mechanical power, bin center (J/min)")
    ax.set_ylabel("28-day cumulative mortality (%)")
    ax.set_title(
        "MP dose-response — scatter (no interpolation between bins)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.text(
        0.5, 0.01,
        "Each marker = bin center; error bar = 95% percentile bootstrap CI. "
        "Bins discretize MP into 20 equal-width intervals on log scale.",
        fontsize=7.5, color="#444444", ha="center", style="italic",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_curves(series, ref_mp, log_x, out):
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    for _, label, color, _marker, x, y, lo, hi in series:
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.plot(x, y, marker="o", lw=2.0, color=color, label=label)
    ax.axvline(ref_mp, color="#7A7A7A", ls="--", lw=1.0, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(
        ref_mp, ymax * 0.97,
        f"  MP = {ref_mp:g} J/min\n  (Costa 2021)",
        color="#7A7A7A", fontsize=8, va="top",
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Mechanical power (J/min)")
    ax.set_ylabel("28-day cumulative mortality (%)")
    ax.set_title(
        "MP dose-response — connected curves (lines are guides to the eye)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.text(
        0.5, 0.01,
        "Lines connect bin centers as visual guides; the model estimates "
        "counterfactual risk only at the 20 binned MP intervals (no "
        "interpolation between bins).",
        fontsize=7.5, color="#444444", ha="center", style="italic",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--out-prefix", required=True, type=Path,
                        help="Path prefix; produces <prefix>_scatter.png and "
                             "<prefix>_curves.png.")
    parser.add_argument("--mp-min", type=float, default=1.0)
    parser.add_argument("--mp-max", type=float, default=40.0)
    parser.add_argument("--reference-mp", type=float, default=17.0)
    parser.add_argument("--log-x", action="store_true",
                        help="Use log-scale x-axis (matches log-MP discretization).")
    args = parser.parse_args()

    z = np.load(args.npz)
    series = _gather(z, args.mp_min, args.mp_max)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_scatter(
        series, args.reference_mp, args.log_x,
        args.out_prefix.with_name(args.out_prefix.name + "_scatter.png"),
    )
    _plot_curves(
        series, args.reference_mp, args.log_x,
        args.out_prefix.with_name(args.out_prefix.name + "_curves.png"),
    )


if __name__ == "__main__":
    main()
