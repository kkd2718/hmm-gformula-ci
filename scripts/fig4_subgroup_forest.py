"""Generate Figure 4: subgroup forest plot of dose-response (high vs low MP bin)
for the proposed VEM-SSM. Yarnell 2023 Fig 4 analog.

Subgroups computed: age (≤65 / >65), sex (M/F), ARDS severity (mild / moderate /
severe), BMI (<30 / ≥30), Charlson tertile (low / mid / high), baseline PEEP
(<8 / ≥8 cmH2O).

For each subgroup, refit VEM-SSM on the subgroup cohort and report risk difference
between a high MP reference bin (default = bin nearest to MP=17 J/min) and a low
MP comparison bin (default = bin nearest median MP).

Output: PNG forest plot + markdown table.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort, ARDSCohort
from src.benchmarks.proposed import VEMSSMBenchmark, VEMConfig
from src.benchmarks._resample import slice_cohort
from src.training.variational_em import TrainingConfig


def _bin_for_mp(cohort: ARDSCohort, mp_value: float) -> int:
    edges = cohort.bin_edges_mp
    centers = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        centers.append(np.sqrt(lo * hi) if lo > 0 else np.nan)
    return int(np.nanargmin(np.abs(np.array(centers) - mp_value)))


def _subgroup_indices(stay_csv_path: Path, stay_ids: np.ndarray,
                      ) -> dict[str, np.ndarray]:
    """Return boolean masks indexed against `stay_ids` order for each subgroup."""
    df = pd.read_csv(stay_csv_path)
    df = df.sort_values(["stay_id", "day_num"]).groupby("stay_id").first().reset_index()
    df = df.set_index("stay_id").reindex(stay_ids)
    age = df["anchor_age"].to_numpy()
    sex = df["gender_M"].to_numpy()
    sev = df["severity"].to_numpy()
    bmi = df["bmi_imputed"].to_numpy()
    cci = df["charlson_index"].to_numpy()
    peep = df["peep"].to_numpy()
    cci_low, cci_hi = np.nanpercentile(cci, [33.3, 66.6])
    return {
        "Age <= 65":           age <= 65,
        "Age > 65":            age > 65,
        "Male":                sex == 1,
        "Female":              sex == 0,
        "Mild ARDS":           sev == "mild",
        "Moderate ARDS":       sev == "moderate",
        "Severe ARDS":         sev == "severe",
        "BMI < 30":            bmi < 30,
        "BMI >= 30":           bmi >= 30,
        "Charlson low":        cci <= cci_low,
        "Charlson mid":        (cci > cci_low) & (cci <= cci_hi),
        "Charlson high":       cci > cci_hi,
        "PEEP < 8":            peep < 8,
        "PEEP >= 8":           peep >= 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--high-mp", type=float, default=17.0)
    parser.add_argument("--low-mp", type=float, default=8.0)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--vem-epochs", type=int, default=200)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--min-n", type=int, default=200)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ARDSConfig(csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t)
    print(f"[load] cohort from {args.csv}")
    full = load_ards_cohort(cfg)
    high_bin = _bin_for_mp(full, args.high_mp)
    low_bin = _bin_for_mp(full, args.low_mp)
    print(f"[ref] high_bin = {high_bin} (~{args.high_mp}), low_bin = {low_bin} (~{args.low_mp})")

    masks = _subgroup_indices(args.csv, full.stay_ids)
    rows = []
    rd_records = []

    def _factory():
        return VEMSSMBenchmark(VEMConfig(training=TrainingConfig(
            n_epochs=args.vem_epochs, learning_rate=1e-2, n_mc_samples=4,
        )))

    for label, mask in masks.items():
        idx = np.where(mask)[0]
        n_sub = len(idx)
        if n_sub < args.min_n:
            print(f"[skip] {label} (n={n_sub} < min_n={args.min_n})")
            rows.append([label, str(n_sub), "-", "-"])
            continue
        sub = slice_cohort(full, idx)
        m = _factory()
        m.fit(sub)
        res = m.dose_response(
            sub, target_bins=[low_bin, high_bin],
            n_bootstrap=args.n_bootstrap, seed=0, refit=False,
        )
        raw = res.risk_raw                               # (2, B)
        rd = (raw[1] - raw[0]) * 100.0
        rd_mean, rd_lo, rd_hi = rd.mean(), np.quantile(rd, 0.025), np.quantile(rd, 0.975)
        rd_records.append((label, n_sub, rd_mean, rd_lo, rd_hi))
        rows.append([
            label, f"{n_sub:,}",
            f"{res.risk_mean[0] * 100:.1f} / {res.risk_mean[1] * 100:.1f}",
            f"{rd_mean:+.1f} ({rd_lo:+.1f}, {rd_hi:+.1f})",
        ])
        print(f"[done] {label}: RD = {rd_mean:+.1f}% ({rd_lo:+.1f}, {rd_hi:+.1f})")

    md_path = args.out_dir / "fig4_subgroup_table.md"
    md_path.write_text(
        "\n".join([
            "# Fig 4 backing data — subgroup risk difference (high vs low MP, VEM-SSM)",
            "",
            f"_high_bin ≈ MP={args.high_mp:g} J/min, low_bin ≈ MP={args.low_mp:g} J/min. "
            f"RD = risk(high) - risk(low) in percentage points. "
            f"Bootstrap B = {args.n_bootstrap} (theta-fixed)._",
            "",
            "| Subgroup | N | Risk (low / high) % | RD (95% CI) |",
            "|---|---|---|---|",
            *("| " + " | ".join(r) + " |" for r in rows),
        ]),
        encoding="utf-8",
    )
    print(f"Wrote {md_path}")

    # Forest plot
    if rd_records:
        labels = [r[0] for r in rd_records]
        means = np.array([r[2] for r in rd_records])
        los = np.array([r[3] for r in rd_records])
        his = np.array([r[4] for r in rd_records])
        y = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(labels) + 1.8), dpi=200)
        ax.errorbar(
            means, y, xerr=[means - los, his - means],
            fmt="o", color="#1F3A5F", capsize=3, lw=1.4,
        )
        ax.axvline(0.0, color="#444444", ls="--", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(
            f"28-day risk difference (high vs low MP, percentage points)"
        )
        ax.set_title("Subgroup risk difference — VEM-SSM g-formula")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        png = args.out_dir / "fig4_subgroup_forest.png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
