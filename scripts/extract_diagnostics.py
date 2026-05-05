"""Aggregate convergence diagnostics across multiple Bayesian fits.

Reads the diagnostics_keys/values arrays saved in state.npz files and
produces a comparison table for Appendix B (R-hat, ESS, divergent count).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_diag(z) -> dict:
    if "diagnostics_keys" not in z.files:
        return {}
    keys = list(z["diagnostics_keys"])
    vals = list(z["diagnostics_values"])
    return {str(k): str(v) for k, v in zip(keys, vals)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-files", nargs="+", required=True, type=Path)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out-md", required=True, type=Path)
    args = p.parse_args()

    md = ["# Posterior diagnostics (Appendix B)", ""]
    md.append("| Method | n params | R-hat max | R-hat p95 | ESS min | ESS median | Divergent |")
    md.append("|---|---|---|---|---|---|---|")
    for path, label in zip(args.state_files, args.labels):
        z = np.load(path, allow_pickle=True)
        d = parse_diag(z)
        if not d:
            md.append(f"| {label} | (no diagnostics in state) | | | | | |")
            continue
        rhat_max = float(d.get("r_hat_max", "nan"))
        rhat_p95 = float(d.get("r_hat_p95", "nan"))
        ess_min = float(d.get("ess_min", "nan"))
        ess_med = float(d.get("ess_median", "nan"))
        n_par = d.get("n_params_with_rhat", "?")
        div = d.get("divergent_count", "?")
        md.append(f"| {label} | {n_par} | {rhat_max:.3f} | {rhat_p95:.3f} | "
                  f"{ess_min:.0f} | {ess_med:.0f} | {div} |")
    md.append("")
    md.append("Conventions: R-hat < 1.05 indicates convergence; ESS bulk ≥ 400 "
              "(2 chains × 1000 sample target) indicates adequate posterior "
              "sample mixing for inference; divergent transitions should be 0 "
              "or near-zero — non-zero divergences flag potential geometric "
              "issues with the posterior surface.")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
