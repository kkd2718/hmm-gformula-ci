"""Generate Table S3: posterior predictive check for the VEM-SSM.

Holds out a 20% test split (cluster on subject_id), fits VEM-SSM on training,
then evaluates one-step-ahead factual hazard prediction on test rows.
Reports Brier score, AUC, and per-day calibration intercepts.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks._resample import slice_cohort
from src.benchmarks.proposed import VEMSSMBenchmark, VEMConfig
from src.training.variational_em import TrainingConfig


def _train_test_indices(subject_ids: np.ndarray, test_frac: float, seed: int,
                        ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_pat = np.unique(subject_ids)
    rng.shuffle(unique_pat)
    n_test_pat = int(round(len(unique_pat) * test_frac))
    test_pat = set(unique_pat[:n_test_pat].tolist())
    test_mask = np.array([s in test_pat for s in subject_ids])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


@torch.no_grad()
def _factual_hazards(model, posterior, cohort) -> np.ndarray:
    """Predict per-(stay,t) factual hazard P(Y_t=1 | observed history)."""
    device = next(model.parameters()).device
    A = cohort.A_bin.to(device)
    L = cohort.L_dyn.to(device)
    V = cohort.C_static.to(device)
    Y = cohort.Y.to(device)
    t_norm = cohort.t_norm.to(device)
    Z, _, _ = posterior.sample_trajectory(A=A, L=L, V=V, Y=Y, n_samples=4)
    Z_mean = Z.mean(dim=0)                                # (N, T, 1)
    N, T, _ = Z_mean.shape
    Z_flat = Z_mean.reshape(N * T, 1)
    A_flat = A.reshape(N * T, -1)
    L_flat = L.reshape(N * T, -1)
    V_rep = V.unsqueeze(1).expand(N, T, V.shape[-1]).reshape(N * T, -1)
    t_flat = t_norm.reshape(N * T, 1)
    logit = model.outcome_logit(Z_flat, A_flat, L_flat, V_rep, t_flat).reshape(N, T)
    return torch.sigmoid(logit).cpu().numpy()


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_p, n_n = len(pos), len(neg)
    all_p = np.concatenate([pos, neg])
    ranks = all_p.argsort().argsort().astype(np.float64) + 1
    rank_pos_sum = ranks[:n_p].sum()
    return float((rank_pos_sum - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--vem-epochs", type=int, default=200)
    parser.add_argument("--vem-z-lag-treatment", action="store_true",
                        help="Option B: A_{t-1} -> Z_t edge in VEM dynamics.")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ARDSConfig(csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t)
    cohort = load_ards_cohort(cfg)
    train_idx, test_idx = _train_test_indices(
        cohort.subject_ids, args.test_frac, args.seed,
    )
    print(f"[split] train={len(train_idx)}, test={len(test_idx)}")
    train_cohort = slice_cohort(cohort, train_idx)
    test_cohort = slice_cohort(cohort, test_idx)

    m = VEMSSMBenchmark(VEMConfig(
        training=TrainingConfig(
            n_epochs=args.vem_epochs, learning_rate=1e-2, n_mc_samples=4,
        ),
        z_depends_on_treatment_lag=args.vem_z_lag_treatment,
    ))
    m.fit(train_cohort)

    p_test = _factual_hazards(m._model, m._posterior, test_cohort)
    Y_test = test_cohort.Y.cpu().numpy().squeeze(-1)
    M_test = test_cohort.at_risk.cpu().numpy().squeeze(-1)

    rows: list[list[str]] = []
    for t in range(args.max_t):
        mask = M_test[:, t] > 0
        if mask.sum() == 0:
            continue
        y_t = Y_test[mask, t]
        p_t = p_test[mask, t]
        brier = float(((p_t - y_t) ** 2).mean())
        auc = _auc(y_t.astype(int), p_t)
        rows.append([
            f"{t}", f"{int(mask.sum()):,}",
            f"{float(y_t.mean()) * 100:.2f}",
            f"{float(p_t.mean()) * 100:.2f}",
            f"{brier:.4f}",
            f"{auc:.3f}" if not np.isnan(auc) else "-",
        ])

    md = [
        "# Table S3. Posterior predictive check on held-out test split (VEM-SSM)",
        "",
        f"_Test split: {args.test_frac * 100:.0f}% of patients (cluster on subject_id), "
        f"n_test = {len(test_idx)}. Per-day factual hazard prediction p_t = "
        f"E_q[P(Y_t=1 | history)] using the posterior mean of Z_t._",
        "",
        "| Day | At-risk N | Observed event % | Predicted hazard % | Brier | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        md.append("| " + " | ".join(row) + " |")

    md_path = args.out_dir / "table_s3_ppc.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
