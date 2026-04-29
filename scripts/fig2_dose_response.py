"""Generate Figure 2: MP dose-response curves for the 3 methods.

Reads `<results_dir>/table2_risks.npz` produced by table2_method_comparison.py
and outputs a PNG with the three estimators overlaid (Yarnell 2023 Fig 2 analog).
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
    "standard_gformula": ("Standard g-formula", "#1F3A5F"),
    "xu_glmm": ("Xu 2024 GLMM g-computation", "#A07A3F"),
    "vem_ssm": ("VEM-SSM g-formula (proposed)", "#B31B1B"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mp-min", type=float, default=2.0)
    parser.add_argument("--mp-max", type=float, default=30.0)
    parser.add_argument("--reference-mp", type=float, default=17.0)
    args = parser.parse_args()

    z = np.load(args.npz)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)
    for key, (label, color) in METHOD_LABELS.items():
        if f"{key}__risk_mean" not in z.files:
            continue
        centers = z[f"{key}__bin_centers_J_min"]
        mean = z[f"{key}__risk_mean"]
        lo = z[f"{key}__risk_ci_low"]
        hi = z[f"{key}__risk_ci_high"]
        mask = np.isfinite(centers) & (centers >= args.mp_min) & (centers <= args.mp_max)
        x = centers[mask]
        order = np.argsort(x)
        x = x[order]
        y = (mean[mask] * 100)[order]
        y_lo = (lo[mask] * 100)[order]
        y_hi = (hi[mask] * 100)[order]
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.15)
        ax.plot(x, y, marker="o", lw=2.0, color=color, label=label)

    ax.axvline(args.reference_mp, color="#7A7A7A", ls="--", lw=1.0, alpha=0.7)
    ax.text(
        args.reference_mp, ax.get_ylim()[1] * 0.95,
        f"  MP = {args.reference_mp:g} J/min\n  (Costa 2021)",
        color="#7A7A7A", fontsize=8, va="top",
    )
    ax.set_xlabel("Mechanical power (J/min)")
    ax.set_ylabel("28-day cumulative mortality (%)")
    ax.set_title(
        "MP dose-response curve — alternative confounding-adjustment specifications"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
