"""Generate Table 3: E-value primary + LOCO secondary sensitivity.

E-value: VanderWeele & Ding (2017) — minimum strength of unmeasured confounding
(on RR scale) needed to explain away the observed risk difference vs the MP=17
reference, computed at each MP bin.

LOCO: leave-one-covariate-out for the proposed VEM-SSM. Each TV covariate is
removed from the cohort, the model refit, and the dose-response point estimate
recorded. Reports the range of point estimates per bin across LOCO refits.

Outputs:
    <out_dir>/table3_sensitivity.md
    <out_dir>/table3_loco.npz
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

from src.data.ards import ARDSConfig, TV_COLS
from src.sensitivity.evalue import evalue_for_rr
from src.sensitivity.loco import run_loco
from src.benchmarks.proposed import VEMSSMBenchmark, VEMConfig
from src.training.variational_em import TrainingConfig


METHOD_LABELS = {
    "standard_gformula": "Standard g-formula",
    "xu_glmm": "Xu 2024 GLMM",
    "vem_ssm": "VEM-SSM (proposed)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path,
                        help="Path to table2_risks.npz from table2_method_comparison.py.")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Path to ards_v31_v4.csv (for LOCO refits).")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--vem-epochs", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-loco", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.npz)
    ref_bin = int(z["reference_bin"][0])

    # ----- E-value table -----
    methods = [m for m in METHOD_LABELS if f"{m}__risk_mean" in z.files]
    centers = z[f"{methods[0]}__bin_centers_J_min"]
    K = len(centers)
    target_bins = np.arange(K)

    rows: list[list[str]] = []
    headers = ["MP bin", "Center (J/min)"] + [
        f"{label} — RR" for k, label in METHOD_LABELS.items() if k in methods
    ] + [
        f"{label} — E-value" for k, label in METHOD_LABELS.items() if k in methods
    ]
    for k in target_bins:
        if k == ref_bin:
            continue
        row = [str(k), f"{centers[k]:.1f}" if np.isfinite(centers[k]) else "-"]
        rrs = []
        for key in methods:
            mean = z[f"{key}__risk_mean"]
            p_t, p_c = float(mean[k]), float(mean[ref_bin])
            if p_c <= 0 or p_t <= 0:
                rrs.append(np.nan); row.append("-"); continue
            rr = p_t / p_c
            rrs.append(rr)
            row.append(f"{rr:.2f}")
        for rr in rrs:
            if not np.isfinite(rr):
                row.append("-"); continue
            row.append(f"{evalue_for_rr(rr):.2f}")
        rows.append(row)

    out_lines = [
        "# Table 3. Sensitivity to unmeasured confounding (E-value) and LOCO covariate exclusion",
        "",
        "## Panel A. E-value per MP bin (VanderWeele & Ding 2017)",
        "",
        f"_Reference: bin {ref_bin} (≈ {centers[ref_bin]:.1f} J/min). RR = "
        f"P(Y|MP=bin) / P(Y|MP=ref). E-value = min unmeasured-confounder RR "
        f"required to explain the observed RR away._",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out_lines.append("| " + " | ".join(row) + " |")

    # ----- LOCO panel for VEM-SSM -----
    if not args.skip_loco:
        print("[LOCO] running VEM-SSM refits per excluded TV covariate ...")
        base_cfg = ARDSConfig(
            csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t,
        )

        def factory():
            return VEMSSMBenchmark(VEMConfig(training=TrainingConfig(
                n_epochs=args.vem_epochs, learning_rate=1e-2, n_mc_samples=4,
            )))

        loco = run_loco(
            method_factory=factory,
            base_cfg=base_cfg,
            target_bins=list(target_bins),
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            refit=False,
            tv_cols=tuple(TV_COLS),
            static_cols=(),
        )
        # Save raw arrays for downstream inspection.
        flat = {}
        for col, res in loco.by_excluded_tv.items():
            flat[f"loco_tv__{col}__risk_mean"] = res.risk_mean
            flat[f"loco_tv__{col}__risk_ci_low"] = res.risk_ci_low
            flat[f"loco_tv__{col}__risk_ci_high"] = res.risk_ci_high
        np.savez(args.out_dir / "table3_loco.npz", **flat)

        # Range of risk_mean across LOCO refits per bin.
        means = np.stack([
            res.risk_mean for res in loco.by_excluded_tv.values()
        ])  # (n_tv, K)
        loco_lo = means.min(axis=0)
        loco_hi = means.max(axis=0)
        full = z["vem_ssm__risk_mean"]

        out_lines += [
            "",
            "## Panel B. LOCO sensitivity for VEM-SSM (TV covariate exclusion)",
            "",
            f"_VEM-SSM refit excluding each of {len(loco.by_excluded_tv)} TV covariates "
            f"({', '.join(loco.by_excluded_tv.keys())}). Reports range of point-estimate "
            f"28-day risk per bin across refits, vs full-covariate primary._",
            "",
            "| MP bin | Center (J/min) | Full primary (%) | LOCO range (%) |",
            "|---|---|---|---|",
        ]
        for k in target_bins:
            c = f"{centers[k]:.1f}" if np.isfinite(centers[k]) else "-"
            out_lines.append(
                f"| {k} | {c} | {full[k] * 100:.1f} | "
                f"[{loco_lo[k] * 100:.1f}, {loco_hi[k] * 100:.1f}] |"
            )

    md_path = args.out_dir / "table3_sensitivity.md"
    md_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
