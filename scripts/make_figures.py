"""Manuscript figures (Fig 3 dose-response + Fig 4 subgroup forest).

Reads risk_*.npz outputs and produces publication-ready figures.

Fig 3A: 4-method dose-response (Standard, Xu, K=1, K=5) overlaid
Fig 3B: K=5 dose-response with credible band, Costa 2021 cutoff annotation
Fig 4 : Subgroup forest plot of (high MP - low MP) risk difference
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Lazy-import matplotlib (not always installed)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:
    print("matplotlib not installed; install with: pip install matplotlib")
    sys.exit(1)


def fig3_dose_response(args):
    R = Path(args.results_dir)
    methods = {
        "Standard": (R / "standard_v2" / "table2_risks.npz", "standard_gformula__"),
        "Xu Bayesian": (R / "bayesian_main_v2" / "xu_bayesian_risks.npz", ""),
        "K=1 NICE": (R / "bayesian_jax" / "fre_nice_K1_risks.npz", ""),
        "K=5 FRE-NICE": (R / "bayesian_jax" / "fre_nice_K5_risks.npz", ""),
    }
    colors = {"Standard": "#666666", "Xu Bayesian": "#FF7F0E",
              "K=1 NICE": "#1F77B4", "K=5 FRE-NICE": "#D62728"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    centers_main = None
    for label, (path, prefix) in methods.items():
        if not path.exists():
            print(f"[fig3] missing {path}, skipping {label}")
            continue
        z = np.load(path)
        rk = f"{prefix}risk_mean" if prefix else "risk_mean"
        rl = f"{prefix}risk_ci_low" if prefix else "risk_ci_low"
        rh = f"{prefix}risk_ci_high" if prefix else "risk_ci_high"
        rc = f"{prefix}bin_centers_J_min" if prefix else "bin_centers_J_min"
        if rk not in z.files:
            print(f"[fig3] {path} has no {rk}; keys: {z.files}")
            continue
        rm = z[rk] * 100
        lo = z[rl] * 100
        hi = z[rh] * 100
        c = z[rc]
        if centers_main is None:
            centers_main = c
        ax1.plot(c, rm, "-o", label=label, color=colors[label],
                 linewidth=1.8, markersize=4)
        if "K=5" in label:
            ax2.plot(c, rm, "-", color=colors[label], linewidth=2.0, label="Posterior mean")
            ax2.fill_between(c, lo, hi, color=colors[label], alpha=0.25,
                             label="95% credible interval")
    from matplotlib.ticker import FixedLocator, FixedFormatter
    xticks = [1, 3, 10, 17, 30]
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.axvline(17.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Mechanical power (J/min)")
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in xticks]))
        ax.xaxis.set_minor_locator(FixedLocator([0.5, 0.7, 2, 5, 7, 20]))
        ax.xaxis.set_minor_formatter(FixedFormatter(["", "", "", "", "", ""]))
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("28-day cumulative incidence (%)")
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.92)
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.92)
    # Panel labels OUTSIDE the axes (above, top-left) to avoid overlap with legend
    for ax, label in [(ax1, "(A)"), (ax2, "(B)")]:
        ax.text(-0.02, 1.04, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold",
                va="bottom", ha="left")
    # Costa cutoff: simple text label above the dashed line, no arrow
    for ax in (ax1, ax2):
        ymax = ax.get_ylim()[1]
        ax.text(17, ymax*0.99, "Costa 2021 cutoff",
                ha="center", va="top", fontsize=8.5, color="#444444",
                rotation=90,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="none", alpha=0.85))
    plt.tight_layout()
    out = Path(args.out_dir) / "fig3_dose_response.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  wrote {out}")


def fig4_subgroup_forest(args):
    R = Path(args.results_dir)
    sub = R / "subgroup" / "subgroup_K5.npz"
    if not sub.exists():
        print(f"[fig4] missing {sub}")
        return
    z = np.load(sub)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["names"]]
    rd = z["rd"]; rd_lo = z["rd_lo"]; rd_hi = z["rd_hi"]; ns = z["ns"]
    # Order: severity, age, BMI (obesity), Charlson (high-comorbidity)
    order = ["mild", "moderate", "severe",
             "age_lt65", "age_geq65",
             "non_obese_lt30", "obese_geq30",
             "charlson_lt5", "charlson_geq5"]
    idx = [names.index(n) for n in order if n in names]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(idx))
    # Subgroup cutoffs use clinical reference thresholds:
    #   Age ≥ 65 yr (older adult, WHO)
    #   BMI ≥ 30 (Class I obesity, WHO non-Asian)
    #   Charlson ≥ 5 (high comorbidity load)
    label_pretty = {
        "mild": "Mild ARDS", "moderate": "Moderate ARDS", "severe": "Severe ARDS",
        "age_lt65": "Age < 65 yr", "age_geq65": "Age ≥ 65 yr",
        "non_obese_lt30": "BMI < 30 (non-obese)", "obese_geq30": "BMI ≥ 30 (obese)",
        "charlson_lt5": "Charlson < 5", "charlson_geq5": "Charlson ≥ 5",
    }
    for i, j in enumerate(idx):
        ax.errorbar(rd[j], y[i],
                    xerr=[[rd[j] - rd_lo[j]], [rd_hi[j] - rd[j]]],
                    fmt="o", color="#D62728", capsize=3, markersize=6)
        # RD with 95% CI displayed at right edge
        ax.text(rd_hi[j] + 2.0, y[i],
                f"{rd[j]:+.1f} ({rd_lo[j]:+.1f}, {rd_hi[j]:+.1f})",
                va="center", fontsize=9, color="#333333", family="monospace")
    ax.set_yticks(y)
    ax.set_yticklabels([label_pretty.get(order[i], order[i]) for i in range(len(idx))])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Risk difference (% points): high MP − low MP")
    # Extend xlim to fit RD text labels on the right
    cur_xlim = ax.get_xlim()
    ax.set_xlim(cur_xlim[0], cur_xlim[1] + 22)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = Path(args.out_dir) / "fig4_subgroup_forest.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results", type=Path)
    p.add_argument("--out-dir", default="results/figures", type=Path)
    p.add_argument("--figs", nargs="+", default=["3", "4"],
                   choices=["3", "4"])
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if "3" in args.figs:
        fig3_dose_response(args)
    if "4" in args.figs:
        fig4_subgroup_forest(args)


if __name__ == "__main__":
    main()
