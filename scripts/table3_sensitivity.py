"""Generate Table 3: E-value (3 methods) + LOCO + grouped exclusion (VEM-SSM only).

Panel A — E-value (VanderWeele & Ding 2017):
    Per-bin RR vs MP=17 reference for each of the 3 methods, with E-value
    quantifying minimum unmeasured-confounding strength to nullify the result.

Panel B — VEM-SSM LOCO + grouped exclusion:
    Refit VEM-SSM under (i) each TV covariate dropped, (ii) each static
    covariate dropped, (iii) named clinical-scenario covariate groups dropped.
    Reports the range of dose-response point estimates per MP bin across
    refits, vs the full-covariate primary estimate.

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

from src.data.ards import ARDSConfig, TV_COLS, STATIC_COLS
from src.sensitivity.evalue import evalue_for_rr
from src.sensitivity.loco import run_loco, GroupExclusion
from src.benchmarks.proposed import VEMSSMBenchmark, VEMConfig
from src.training.variational_em import TrainingConfig


METHOD_LABELS = {
    "standard_gformula": "Standard parametric g-formula",
    "xu_glmm": "Xu 2024 GLMM",
    "vem_ssm": "VEM-SSM (proposed)",
}

# Pre-specified clinical-scenario covariate groups for grouped exclusion.
CLINICAL_GROUPS = [
    GroupExclusion(
        label="All TV covariates",
        tv_cols=tuple(TV_COLS),
    ),
    GroupExclusion(
        label="All static covariates",
        static_cols=tuple(STATIC_COLS),
    ),
    GroupExclusion(
        label="Oxygenation domain (P/F + PaCO2)",
        tv_cols=("pf_ratio", "paco2"),
    ),
    GroupExclusion(
        label="Hemodynamic domain (HR + MAP)",
        tv_cols=("heart_rate", "map_mmhg"),
    ),
    GroupExclusion(
        label="Metabolic/renal domain (lactate + creatinine)",
        tv_cols=("lactate", "creatinine"),
    ),
]


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
    methods = [m for m in METHOD_LABELS if f"{m}__risk_mean" in z.files]
    centers = z[f"{methods[0]}__bin_centers_J_min"]
    K = len(centers)
    target_bins = list(range(K))

    out_lines: list[str] = [
        "# Table 3. Sensitivity to unmeasured confounding (E-value) and "
        "covariate-set choice (LOCO + grouped exclusion)",
        "",
        "## Panel A. E-value per MP bin (VanderWeele & Ding 2017) — three methods",
        "",
        f"_Reference bin: bin {ref_bin} (≈ {centers[ref_bin]:.1f} J/min, "
        f"Costa 2021 cutoff). RR = P(Y|MP=bin) / P(Y|MP=ref). "
        f"E-value = minimum unmeasured-confounder RR required to nullify the "
        f"observed RR._",
        "",
    ]

    headers = ["MP bin", "Center (J/min)"] + [
        f"{label} — RR" for k, label in METHOD_LABELS.items() if k in methods
    ] + [
        f"{label} — E-value" for k, label in METHOD_LABELS.items() if k in methods
    ]
    out_lines.append("| " + " | ".join(headers) + " |")
    out_lines.append("|" + "|".join(["---"] * len(headers)) + "|")

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
            row.append(f"{evalue_for_rr(rr):.2f}" if np.isfinite(rr) else "-")
        out_lines.append("| " + " | ".join(row) + " |")

    if args.skip_loco:
        md_path = args.out_dir / "table3_sensitivity.md"
        md_path.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"Wrote {md_path} (Panel A only; LOCO skipped)")
        return

    print("[LOCO] running VEM-SSM refits per excluded covariate / group ...")
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
        target_bins=target_bins,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        refit=False,
        tv_cols=tuple(TV_COLS),
        static_cols=tuple(STATIC_COLS),
        groups=CLINICAL_GROUPS,
    )

    flat: dict[str, np.ndarray] = {}
    for col, res in loco.by_excluded_tv.items():
        flat[f"loco_tv__{col}__risk_mean"] = res.risk_mean
        flat[f"loco_tv__{col}__risk_ci_low"] = res.risk_ci_low
        flat[f"loco_tv__{col}__risk_ci_high"] = res.risk_ci_high
    for col, res in loco.by_excluded_static.items():
        flat[f"loco_static__{col}__risk_mean"] = res.risk_mean
        flat[f"loco_static__{col}__risk_ci_low"] = res.risk_ci_low
        flat[f"loco_static__{col}__risk_ci_high"] = res.risk_ci_high
    for label, res in loco.by_excluded_group.items():
        safe = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        flat[f"group__{safe}__risk_mean"] = res.risk_mean
        flat[f"group__{safe}__risk_ci_low"] = res.risk_ci_low
        flat[f"group__{safe}__risk_ci_high"] = res.risk_ci_high
    np.savez(args.out_dir / "table3_loco.npz", **flat)

    full_vem = z["vem_ssm__risk_mean"]

    def _stack_means(d: dict) -> tuple[np.ndarray, list[str]]:
        labels = list(d.keys())
        if not labels:
            return np.zeros((0, K)), []
        return np.stack([d[k].risk_mean for k in labels], axis=0), labels

    tv_means, tv_labels = _stack_means(loco.by_excluded_tv)
    static_means, static_labels = _stack_means(loco.by_excluded_static)
    group_means, group_labels = _stack_means(loco.by_excluded_group)

    out_lines += [
        "",
        "## Panel B. VEM-SSM LOCO + grouped exclusion sensitivity",
        "",
        f"_VEM-SSM refit excluding (i) each of {len(tv_labels)} TV covariates, "
        f"(ii) each of {len(static_labels)} static covariates, "
        f"(iii) {len(group_labels)} clinical-scenario groups. "
        f"Per-bin range = [min, max] of point-estimate 28-day risk across refits "
        f"vs full-covariate primary._",
        "",
        "| MP bin | Center (J/min) | Full primary (%) | TV LOCO range (%) | Static LOCO range (%) | Grouped exclusion range (%) |",
        "|---|---|---|---|---|---|",
    ]
    for k in target_bins:
        c = f"{centers[k]:.1f}" if np.isfinite(centers[k]) else "-"
        full = f"{full_vem[k] * 100:.1f}"
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
        out_lines.append(f"| {k} | {c} | {full} | {tv_rng} | {st_rng} | {gr_rng} |")

    out_lines += [
        "",
        "### Panel B detail — per-exclusion VEM-SSM risk at reference bin and high bin",
        "",
        f"_Reference bin {ref_bin}, max bin {K - 1}. Reports per-refit risk to identify "
        f"which exclusions, if any, drive non-trivial change._",
        "",
        "| Exclusion | Type | Risk @ ref bin (%) | Risk @ max bin (%) |",
        "|---|---|---|---|",
        f"| (full covariate primary) | — | "
        f"{full_vem[ref_bin] * 100:.1f} | {full_vem[-1] * 100:.1f} |",
    ]
    for label, res in loco.by_excluded_tv.items():
        out_lines.append(
            f"| {label} | TV | "
            f"{res.risk_mean[ref_bin] * 100:.1f} | {res.risk_mean[-1] * 100:.1f} |"
        )
    for label, res in loco.by_excluded_static.items():
        out_lines.append(
            f"| {label} | Static | "
            f"{res.risk_mean[ref_bin] * 100:.1f} | {res.risk_mean[-1] * 100:.1f} |"
        )
    for label, res in loco.by_excluded_group.items():
        out_lines.append(
            f"| {label} | Group | "
            f"{res.risk_mean[ref_bin] * 100:.1f} | {res.risk_mean[-1] * 100:.1f} |"
        )

    md_path = args.out_dir / "table3_sensitivity.md"
    md_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
