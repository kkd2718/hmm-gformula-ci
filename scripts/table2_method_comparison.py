"""Generate Table 2 (3-way method comparison: 28-day RD per MP bin).

Runs: StandardGFormula (refit), XuGLMM (refit), VEMSSMBenchmark (theta-fixed).
Outputs:
    <out_dir>/table2_method_comparison.md
    <out_dir>/table2_risks.npz   (risk_mean / CI / raw bootstrap arrays per method)

Designed for batch execution on a CPU/GPU VM. CLI args override defaults so the
same script runs locally on Windows and on a Linux VM.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks import StandardGFormula, XuGLMM, VEMSSMBenchmark, VEMConfig
from src.training.variational_em import TrainingConfig


REFERENCE_MP_J_MIN = 17.0


def find_reference_bin(cohort, mp_value: float = REFERENCE_MP_J_MIN) -> int:
    """Return bin index whose geometric center is closest to `mp_value` J/min."""
    edges = cohort.bin_edges_mp
    centers = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
            centers.append(np.sqrt(lo * hi))
        else:
            centers.append(np.nan)
    centers = np.array(centers)
    return int(np.nanargmin(np.abs(centers - mp_value)))


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path,
                        help="Path to ards_v31_v4.csv (v4 cohort with SOFA).")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--n-bootstrap-cls", type=int, default=100,
                        help="Bootstrap reps for StandardGFormula and XuGLMM (refit).")
    parser.add_argument("--n-bootstrap-vem", type=int, default=100,
                        help="Bootstrap reps for VEM-SSM (theta-fixed).")
    parser.add_argument("--xu-b-draws", type=int, default=200,
                        help="MC draws over b_i posterior for XuGLMM.")
    parser.add_argument("--vem-refit", action="store_true",
                        help="Use refit-bootstrap for VEM-SSM (symmetric with Std/Xu); "
                             "expensive but methodologically symmetric.")
    parser.add_argument("--vem-epochs", type=int, default=200)
    parser.add_argument("--vem-lr", type=float, default=1e-2)
    parser.add_argument("--vem-mc", type=int, default=4)
    parser.add_argument("--reference-mp", type=float, default=REFERENCE_MP_J_MIN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["standard", "xu", "vem"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ARDSConfig(csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t)
    print(f"[load] cohort from {args.csv}")
    cohort = load_ards_cohort(cfg)
    N = cohort.Y.shape[0]
    K = cohort.feature_layout["n_bins"]
    print(f"[load] N stays = {N}, K bins = {K}")

    ref_bin = find_reference_bin(cohort, args.reference_mp)
    target_bins = list(range(K))
    print(f"[ref] reference bin = {ref_bin} (target ~ {args.reference_mp} J/min)")

    results: dict[str, dict] = {}

    if "standard" not in args.skip:
        print("[standard_gformula] fitting + refit-bootstrap ...")
        m_std = StandardGFormula()
        res_std = m_std.dose_response(
            cohort, target_bins=target_bins,
            n_bootstrap=args.n_bootstrap_cls, seed=args.seed, refit=True,
        )
        results["standard_gformula"] = {
            "risk_mean": res_std.risk_mean,
            "risk_ci_low": res_std.risk_ci_low,
            "risk_ci_high": res_std.risk_ci_high,
            "risk_raw": res_std.risk_raw,
            "bin_centers_J_min": np.array(res_std.bin_centers_J_min),
        }
        print(f"[standard_gformula] done")

    if "xu" not in args.skip:
        print("[xu_glmm] fitting + refit-bootstrap ...")
        m_xu = XuGLMM(n_b_draws=args.xu_b_draws)
        res_xu = m_xu.dose_response(
            cohort, target_bins=target_bins,
            n_bootstrap=args.n_bootstrap_cls, seed=args.seed + 1, refit=True,
        )
        results["xu_glmm"] = {
            "risk_mean": res_xu.risk_mean,
            "risk_ci_low": res_xu.risk_ci_low,
            "risk_ci_high": res_xu.risk_ci_high,
            "risk_raw": res_xu.risk_raw,
            "bin_centers_J_min": np.array(res_xu.bin_centers_J_min),
        }
        print(f"[xu_glmm] done")

    if "vem" not in args.skip:
        vem_cfg = VEMConfig(training=TrainingConfig(
            n_epochs=args.vem_epochs, learning_rate=args.vem_lr,
            n_mc_samples=args.vem_mc,
        ))
        m_vem = VEMSSMBenchmark(vem_cfg)
        # Pre-fit on full cohort (always) to have a parameter snapshot for diagnostics.
        if not args.vem_refit:
            print("[vem_ssm] fitting (theta on full cohort) ...")
            m_vem.fit(cohort)
        else:
            # For refit-bootstrap, still do an initial fit so we can save snapshot.
            m_vem.fit(cohort)
        param_snap = m_vem.fitted_param_snapshot()
        import json as _json
        (args.out_dir / "vem_param_snapshot.json").write_text(
            _json.dumps(param_snap, indent=2), encoding="utf-8",
        )
        print(f"[vem_ssm] saved parameter snapshot")

        if args.vem_refit:
            print("[vem_ssm] dose-response (refit-bootstrap, symmetric with Std/Xu) ...")
            res_vem = m_vem.dose_response(
                cohort, target_bins=target_bins,
                n_bootstrap=args.n_bootstrap_vem, seed=args.seed + 2, refit=True,
            )
        else:
            print("[vem_ssm] dose-response (theta-fixed bootstrap) ...")
            res_vem = m_vem.dose_response(
                cohort, target_bins=target_bins,
                n_bootstrap=args.n_bootstrap_vem, seed=args.seed + 2, refit=False,
            )
        results["vem_ssm"] = {
            "risk_mean": res_vem.risk_mean,
            "risk_ci_low": res_vem.risk_ci_low,
            "risk_ci_high": res_vem.risk_ci_high,
            "risk_raw": res_vem.risk_raw,
            "bin_centers_J_min": np.array(res_vem.bin_centers_J_min),
        }
        print(f"[vem_ssm] done")

    # Save raw arrays for downstream figures.
    npz_path = args.out_dir / "table2_risks.npz"
    flat = {
        f"{m}__{k}": v for m, d in results.items() for k, v in d.items()
    }
    flat["target_bins"] = np.array(target_bins, dtype=int)
    flat["reference_bin"] = np.array([ref_bin], dtype=int)
    np.savez(npz_path, **flat)
    print(f"[save] {npz_path}")

    # Build Table 2 markdown.
    methods_order = [
        ("standard_gformula", "Standard g-formula"),
        ("xu_glmm", "Xu 2024 GLMM g-computation"),
        ("vem_ssm", "VEM-SSM g-formula (proposed)"),
    ]
    headers = ["MP bin", "MP center (J/min)"] + [
        f"{label} — risk % (95% CI)" for _, label in methods_order if _ in results
    ] + [
        f"{label} — RD vs MP={args.reference_mp:g} (95% CI)"
        for _, label in methods_order if _ in results
    ]
    rows: list[list[str]] = []
    centers = (
        results[next(iter(results))]["bin_centers_J_min"]
        if results else np.zeros(K)
    )
    for k in target_bins:
        row = [f"{k}", f"{centers[k]:.1f}" if np.isfinite(centers[k]) else "-"]
        for key, _ in methods_order:
            if key not in results:
                continue
            r = results[key]
            row.append(
                f"{fmt_pct(r['risk_mean'][k])} "
                f"({fmt_pct(r['risk_ci_low'][k])}-{fmt_pct(r['risk_ci_high'][k])})"
            )
        for key, _ in methods_order:
            if key not in results:
                continue
            r = results[key]
            raw = r["risk_raw"]
            rd_raw = raw[k] - raw[ref_bin]
            rd_mean = rd_raw.mean()
            rd_lo = np.quantile(rd_raw, 0.025)
            rd_hi = np.quantile(rd_raw, 0.975)
            row.append(f"{fmt_pct(rd_mean):>5} ({fmt_pct(rd_lo)}, {fmt_pct(rd_hi)})")
        rows.append(row)

    out = [
        "# Table 2. 28-day mortality risk by MP bin: alternative confounding-adjustment specifications",
        "",
        f"_Reference bin: bin {ref_bin} (≈ {args.reference_mp:g} J/min, Costa 2021 cutoff). "
        f"Risks are 28-day cumulative incidence (%); RD is risk difference vs reference. "
        f"Bootstrap: B = {args.n_bootstrap_cls} (refit) for Standard / Xu, "
        f"B = {args.n_bootstrap_vem} (theta-fixed) for VEM-SSM._",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")

    md_path = args.out_dir / "table2_method_comparison.md"
    md_path.write_text("\n".join(out), encoding="utf-8")
    print(f"[save] {md_path}")


if __name__ == "__main__":
    main()
