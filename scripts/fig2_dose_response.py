"""Generate Figure 2: MP dose-response — scatter with 95% CI error bars.

Reads `<results_dir>/table2_risks.npz` and plots per-bin point estimates with
95% CI vertical error bars for each of the three methods. **No connecting
lines** — bins are MP intervals (not points), so smooth interpolation between
bin centers would misrepresent the discrete counterfactual estimand
(Costa 2021 AJRCCM / Serpa Neto 2018 ICM convention).
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mp-min", type=float, default=1.0)
    parser.add_argument("--mp-max", type=float, default=40.0)
    parser.add_argument("--reference-mp", type=float, default=17.0)
    parser.add_argument("--log-x", action="store_true",
                        help="Use log-scale x-axis (matches log-MP discretization).")
    args = parser.parse_args()

    z = np.load(args.npz)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)

    n_methods = sum(1 for k in METHOD_LABELS if f"{k}__risk_mean" in z.files)
    offset_step = 0.04
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * offset_step

    i = 0
    for key, (label, color, marker) in METHOD_LABELS.items():
        if f"{key}__risk_mean" not in z.files:
            continue
        centers = z[f"{key}__bin_centers_J_min"]
        mean = z[f"{key}__risk_mean"]
        lo = z[f"{key}__risk_ci_low"]
        hi = z[f"{key}__risk_ci_high"]
        mask = np.isfinite(centers) & (centers >= args.mp_min) & (centers <= args.mp_max)
        x = centers[mask].astype(float)
        x_off = x * (1.0 + offsets[i]) if args.log_x else x + offsets[i] * 5.0
        y = (mean[mask] * 100)
        y_lo = (lo[mask] * 100)
        y_hi = (hi[mask] * 100)
        order = np.argsort(x)
        ax.errorbar(
            x_off[order], y[order],
            yerr=[y[order] - y_lo[order], y_hi[order] - y[order]],
            fmt=marker, color=color, ecolor=color, lw=1.0,
            markersize=5, capsize=3, label=label, alpha=0.85,
        )
        i += 1

    ax.axvline(args.reference_mp, color="#7A7A7A", ls="--", lw=1.0, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(
        args.reference_mp, ymax * 0.97,
        f"  MP = {args.reference_mp:g} J/min\n  (Costa 2021)",
        color="#7A7A7A", fontsize=8, va="top",
    )
    if args.log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Mechanical power, bin center (J/min)")
    ax.set_ylabel("28-day cumulative mortality (%)")
    ax.set_title(
        "MP dose-response — alternative confounding-adjustment specifications"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.text(
        0.5, 0.01,
        "Each marker = bin center; error bar = 95% percentile bootstrap CI. "
        "Bins discretize MP into 20 equal-width intervals on log scale; "
        "no interpolation between bins.",
        fontsize=7.5, color="#444444", ha="center", style="italic",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
