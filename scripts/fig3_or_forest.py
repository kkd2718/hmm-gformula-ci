"""Generate Figure 3: forest plot of OR for 28-day mortality vs MP=17 J/min reference.

Reads `<results_dir>/table2_risks.npz` and computes per-bin log-OR estimates with
95% percentile CIs over bootstrap replicates (Yarnell 2023 Fig 3 analog).
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


def _odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p / (1.0 - p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference-mp", type=float, default=17.0)
    args = parser.parse_args()

    z = np.load(args.npz)
    ref_bin = int(z["reference_bin"][0])
    methods = [m for m in METHOD_LABELS if f"{m}__risk_raw" in z.files]
    n_methods = len(methods)
    if n_methods == 0:
        print("No results in npz", file=sys.stderr); sys.exit(1)

    centers = z[f"{methods[0]}__bin_centers_J_min"]
    K = len(centers)
    bins = np.arange(K)

    fig, ax = plt.subplots(figsize=(8.5, 0.32 * K + 1.8), dpi=200)
    offset_step = 0.25
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * offset_step

    for i, key in enumerate(methods):
        label, color = METHOD_LABELS[key]
        raw = z[f"{key}__risk_raw"]                   # (K, B)
        odds_ratio = _odds(raw) / _odds(raw[ref_bin])[None, :]
        log_or = np.log(odds_ratio)
        mean_or = np.exp(log_or.mean(axis=1))
        lo = np.exp(np.quantile(log_or, 0.025, axis=1))
        hi = np.exp(np.quantile(log_or, 0.975, axis=1))
        y = bins + offsets[i]
        ax.errorbar(
            mean_or, y, xerr=[mean_or - lo, hi - mean_or],
            fmt="o", color=color, lw=1.5, markersize=4, capsize=3, label=label,
        )

    ax.axvline(1.0, color="#444444", ls="--", lw=0.9)
    ax.axhline(ref_bin, color="#B31B1B", ls=":", lw=0.8, alpha=0.5)
    ax.set_yticks(bins)
    ax.set_yticklabels(
        [f"{k} ({centers[k]:.1f})" if np.isfinite(centers[k]) else f"{k}" for k in bins]
    )
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(f"OR (vs MP ≈ {args.reference_mp:g} J/min, log scale)")
    ax.set_ylabel("MP bin (J/min center)")
    ax.set_title(f"OR for 28-day mortality vs MP = {args.reference_mp:g} J/min reference")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
