"""Diagnose VEM training stability across bootstrap cohorts.

For 3 bootstrap replicates × hyperparameter grid:
    n_epochs ∈ {50, 100, 200}
    lr ∈ {1e-3, 5e-3, 1e-2}
    smoothness_lambda ∈ {0.0, 0.02}
    seed ∈ {0, 1, 2}

For each cell, fit VEM on bootstrap cohort and report:
- Final ELBO
- ELBO at epoch 25, 50, 100, 200
- Plateau epoch (relative change < 0.001 over 25-epoch window)
- Natural-course mortality estimate

Output: stability_grid.csv + stability_curves.png (loss curves overlaid).
"""
from __future__ import annotations
import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks._resample import cluster_bootstrap_indices, slice_cohort
from src.benchmarks import VEMSSMBenchmark, VEMConfig
from src.training.variational_em import TrainingConfig
from scripts.sanity_check import natural_course_simulation


def _check_plateau(elbo: list[float], window: int = 25, rel_tol: float = 1e-3) -> int | None:
    for t in range(window, len(elbo)):
        denom = max(abs(elbo[t]), 1e-6)
        if abs(elbo[t] - elbo[t - window]) / denom < rel_tol:
            return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-boot-cohorts", type=int, default=3)
    parser.add_argument("--n-epochs-grid", type=int, nargs="+", default=[200])
    parser.add_argument("--lr-grid", type=float, nargs="+", default=[1e-3, 5e-3, 1e-2])
    parser.add_argument("--smooth-grid", type=float, nargs="+", default=[0.0, 0.02])
    parser.add_argument("--seed-grid", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t,
    ))

    # Generate boot cohorts (same for all hyperparam cells)
    rng = np.random.default_rng(42)
    boot_cohorts = []
    for b in range(args.n_boot_cohorts):
        idx = cluster_bootstrap_indices(full.subject_ids, rng)
        boot_cohorts.append((b, slice_cohort(full, idx)))

    rows = []
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    color_idx = 0

    n_cells = (
        args.n_boot_cohorts
        * len(args.n_epochs_grid) * len(args.lr_grid)
        * len(args.smooth_grid) * len(args.seed_grid)
    )
    print(f"Running {n_cells} cells ...")
    cell = 0
    for b_idx, boot_cohort in boot_cohorts:
        for n_ep, lr, sm, seed in product(
            args.n_epochs_grid, args.lr_grid, args.smooth_grid, args.seed_grid,
        ):
            cell += 1
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg = VEMConfig(
                training=TrainingConfig(
                    n_epochs=n_ep, learning_rate=lr,
                    smoothness_lambda=sm, n_mc_samples=4, grad_clip=5.0,
                    print_every=10000, early_stop_patience=10**6,
                    early_stop_tol=0.0,
                ),
                z_depends_on_treatment_lag=True,
            )
            m = VEMSSMBenchmark(cfg)
            print(f"[{cell}/{n_cells}] boot={b_idx} ep={n_ep} lr={lr} smooth={sm} seed={seed} ...")
            try:
                m.fit(boot_cohort)
                history = m.last_history
                elbo = history.elbo
                plateau = _check_plateau(elbo)
                nc = natural_course_simulation(
                    m._model, boot_cohort, n_mc_subjects=boot_cohort.Y.shape[0],
                    seed=seed,
                )
                rows.append({
                    "boot": b_idx, "epochs": n_ep, "lr": lr, "smooth": sm, "seed": seed,
                    "elbo_e25": elbo[24] if len(elbo) > 24 else float("nan"),
                    "elbo_e50": elbo[49] if len(elbo) > 49 else float("nan"),
                    "elbo_e100": elbo[99] if len(elbo) > 99 else float("nan"),
                    "elbo_final": elbo[-1],
                    "plateau_epoch": plateau if plateau is not None else -1,
                    "natural_course_pct": nc * 100,
                })
                # Plot loss curve
                label = f"b{b_idx}/lr{lr}/sm{sm}/s{seed}"
                ax.plot(elbo, lw=0.9, alpha=0.7, label=label)
                color_idx += 1
            except Exception as e:
                print(f"  FAIL: {e!r}")
                rows.append({
                    "boot": b_idx, "epochs": n_ep, "lr": lr, "smooth": sm, "seed": seed,
                    "elbo_e25": float("nan"), "elbo_e50": float("nan"),
                    "elbo_e100": float("nan"), "elbo_final": float("nan"),
                    "plateau_epoch": -1, "natural_course_pct": float("nan"),
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "stability_grid.csv", index=False)
    print(f"\nWrote {args.out_dir / 'stability_grid.csv'}")
    print(df.to_string())

    ax.set_xlabel("epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("VEM ELBO trajectories across bootstrap cohorts × hyperparams")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out_dir / "stability_curves.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out_dir / 'stability_curves.png'}")


if __name__ == "__main__":
    main()
