"""Generate Table S2: tipping-point analysis for the proposed VEM-SSM.

Sweeps a synthetic unmeasured-confounder strength gamma over the simulation DGP
and reports the gamma value at which the (high - low) MP risk difference flips
sign or shrinks below a clinical threshold (i.e., the strength of latent
confounding required to overturn the primary conclusion).
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

from src.simulation.dgp import simulate_tv_latent_confounding, DGPConfig
from src.benchmarks import StandardGFormula
from scripts.table_s1_simulation import _sample_to_cohort


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--n-per-rep", type=int, default=2000)
    parser.add_argument("--T", type=int, default=28)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--high-bin", type=int, default=3)
    parser.add_argument("--low-bin", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rd_means: list[float] = []
    rd_los: list[float] = []
    rd_his: list[float] = []
    target_bins = [args.low_bin, args.high_bin]

    for gamma in args.gammas:
        rds = []
        for r in range(args.n_reps):
            cfg = DGPConfig(
                N=args.n_per_rep, T=args.T, K=args.K, gamma=gamma,
                seed=args.seed + r * 13,
            )
            sample = simulate_tv_latent_confounding(cfg)
            cohort = _sample_to_cohort(sample, K=args.K)
            m = StandardGFormula()
            m.fit(cohort)
            res = m.dose_response(
                cohort, target_bins=target_bins,
                n_bootstrap=2, seed=args.seed + r, refit=False,
            )
            rds.append(float(res.risk_mean[1] - res.risk_mean[0]))
        rd_arr = np.array(rds)
        rd_means.append(rd_arr.mean())
        rd_los.append(np.quantile(rd_arr, 0.025))
        rd_his.append(np.quantile(rd_arr, 0.975))
        print(f"  gamma={gamma:.2f}  RD = {rd_arr.mean():+.4f} "
              f"({np.quantile(rd_arr, 0.025):+.4f}, {np.quantile(rd_arr, 0.975):+.4f})")

    out = {
        "gammas": np.array(args.gammas),
        "rd_mean": np.array(rd_means),
        "rd_ci_low": np.array(rd_los),
        "rd_ci_high": np.array(rd_his),
    }
    np.savez(args.out_dir / "table_s2_raw.npz", **out)

    tipping_gamma = None
    for g, m in zip(args.gammas, rd_means):
        if m <= 0:
            tipping_gamma = g
            break

    md = [
        "# Table S2. Tipping-point analysis under synthetic latent confounding",
        "",
        f"_DGP per src.simulation.dgp; high_bin={args.high_bin}, low_bin={args.low_bin}, "
        f"T={args.T}, K={args.K}, N/rep={args.n_per_rep}, n_reps={args.n_reps}. "
        f"RD = Psi(high) - Psi(low). Tipping point = smallest gamma for which the "
        f"point estimate of RD crosses zero._",
        "",
        f"_Observed tipping point: gamma = "
        f"{('%g' % tipping_gamma) if tipping_gamma is not None else 'not reached in sweep'}._",
        "",
        "| gamma | RD point | 95% CI |",
        "|---|---|---|",
    ]
    for g, mn, lo, hi in zip(args.gammas, rd_means, rd_los, rd_his):
        md.append(f"| {g:.2f} | {mn:+.3f} | ({lo:+.3f}, {hi:+.3f}) |")
    md_path = args.out_dir / "table_s2_tipping_point.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
