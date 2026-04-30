"""Generate side-by-side causal DAGs for VEM-SSM g-formula option A and B.

Option A (conservative): Z_t depends only on Z_{t-1}, L_{t-1}, V.
    Z_t represents intrinsic patient vulnerability, exogenous to treatment.

Option B (full): adds A_{t-1} -> Z_t edge.
    Z_t represents endogenous severity influenced by past treatment as well.

Both share: V influences every Z_t, L_t, A_t, Y_t (drawn selectively for clarity);
within-timestep ordering Z_t -> L_t -> A_t -> Y_t with Z_t -> Y_t and L_t -> Y_t direct;
cross-time AR on Z, L, A; A_{t-1} -> L_t (treatment-confounder feedback).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COL_V = "#7A7A7A"
COL_Z = "#B31B1B"
COL_L = "#1F3A5F"
COL_A = "#2E7A4D"
COL_Y = "#000000"
COL_HIGHLIGHT = "#FF8C00"


def _node(ax, x, y, label, color, latent=False, outcome=False, radius=0.30):
    fc = "white" if not outcome else color
    text_color = "white" if outcome else color
    ls = "dashed" if latent else "solid"
    ax.add_patch(Circle(
        (x, y), radius=radius, facecolor=fc, edgecolor=color,
        linewidth=2.0, linestyle=ls, zorder=3,
    ))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color, zorder=4)


def _arrow(ax, p_from, p_to, color="#444444", curve=0.0, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p_from, p_to,
        arrowstyle="-|>,head_length=10,head_width=7", mutation_scale=1.0,
        connectionstyle=f"arc3,rad={curve}",
        color=color, lw=lw, linestyle=ls, zorder=2,
        shrinkA=14, shrinkB=14,
    ))


def _draw_panel(ax, option: str, x_off: float = 0.0):
    y_Z, y_L, y_A, y_Y, y_V = 7.2, 5.4, 3.6, 1.5, 4.5
    x_tm1, x_t, x_tp1 = 2.5 + x_off, 6.0 + x_off, 9.5 + x_off
    x_V = 0.3 + x_off

    _node(ax, x_V, y_V, "V", COL_V)

    for x, suf in [(x_tm1, "{t-1}"), (x_t, "t"), (x_tp1, "{t+1}")]:
        _node(ax, x, y_Z, rf"$Z_{suf}$", COL_Z, latent=True)
        _node(ax, x, y_L, rf"$L_{suf}$", COL_L)
        _node(ax, x, y_A, rf"$A_{suf}$", COL_A)
        _node(ax, x, y_Y, rf"$Y_{suf}$", COL_Y, outcome=True)

    # V -> Z at every timestep (baseline conditioning, dotted)
    for x in [x_tm1, x_t, x_tp1]:
        _arrow(ax, (x_V, y_V), (x, y_Z), color=COL_V, curve=0.30, lw=1.0, ls=":")
    # V -> Y_{t+1} (illustrative; full V->all would clutter)
    _arrow(ax, (x_V, y_V), (x_tp1, y_Y), color=COL_V, curve=-0.45, lw=1.0, ls=":")

    # Within-timestep: Z->L, L->A, Z->Y, L->Y, A->Y
    for x in [x_tm1, x_t, x_tp1]:
        _arrow(ax, (x, y_Z), (x, y_L), color=COL_Z, lw=1.6)
        _arrow(ax, (x, y_L), (x, y_A), color=COL_L, lw=1.6)
        _arrow(ax, (x, y_Z), (x, y_Y), color=COL_Z, curve=0.45, lw=1.6)
        _arrow(ax, (x, y_L), (x, y_Y), color=COL_L, lw=1.6)
        _arrow(ax, (x, y_A), (x, y_Y), color=COL_A, lw=2.0)

    # Cross-timestep
    for x_a, x_b in [(x_tm1, x_t), (x_t, x_tp1)]:
        _arrow(ax, (x_a, y_Z), (x_b, y_Z), color=COL_Z, lw=1.8)
        _arrow(ax, (x_a, y_L), (x_b, y_Z), color=COL_L, curve=-0.18, lw=1.3)
        _arrow(ax, (x_a, y_L), (x_b, y_L), color=COL_L, lw=1.6)
        _arrow(ax, (x_a, y_A), (x_b, y_L), color=COL_A, curve=-0.18, lw=1.4)
        _arrow(ax, (x_a, y_A), (x_b, y_A), color=COL_A, lw=1.3)

        # ★ Option B only: A_{t-1} -> Z_t
        if option == "B":
            _arrow(ax, (x_a, y_A), (x_b, y_Z), color=COL_HIGHLIGHT, curve=0.30, lw=2.4)

    for x, lab in [(x_tm1, "Day t−1"), (x_t, "Day t"), (x_tp1, "Day t+1")]:
        ax.text(x, 8.3, lab, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#444444")

    title_lines = {
        "A": ("Option A — conservative",
              r"$Z_t$ exogenous to treatment ($Z_t \perp A_{t-1}$ given $Z_{t-1}, L_{t-1}, V$)"),
        "B": ("Option B — full (proposed primary)",
              r"$Z_t$ endogenous: includes $A_{t-1} \to Z_t$ (orange)"),
    }[option]
    ax.text((x_tm1 + x_tp1) / 2, 9.4, title_lines[0],
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text((x_tm1 + x_tp1) / 2, 8.95, title_lines[1],
            ha="center", va="center", fontsize=9, color="#444444", style="italic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 8.0), dpi=200)
    for ax in (axA, axB):
        ax.set_xlim(-0.5, 11.0)
        ax.set_ylim(0.2, 9.8)
        ax.axis("off")

    _draw_panel(axA, "A")
    _draw_panel(axB, "B")

    fig.suptitle(
        "Proposed VEM-SSM g-formula: two latent dynamics specifications",
        fontsize=14, fontweight="bold", y=0.99,
    )

    legend = (
        "Dashed circle = unmeasured (latent); solid = observed.   "
        "Dotted edges from V are baseline conditioning (drawn selectively to reduce clutter; "
        "V influences every Z, L, A, Y in both panels).   "
        r"Orange edge in panel B = additional $A_{t-1} \to Z_t$ dependency tested as primary spec."
    )
    fig.text(0.5, 0.03, legend, fontsize=9, color="#222222", ha="center")

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
