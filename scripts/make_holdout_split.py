"""Subject-stratified random 80/20 split for held-out PPC.

Saves train_subj_ids.npy and holdout_subj_ids.npy with subject indices
(integers in [0, n_subjects)). Same indexing scheme as benchmark fit code.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_ards_cohort(ARDSConfig(csv_path=args.csv, n_bins=args.n_bins))
    _, inv = np.unique(cohort.subject_ids, return_inverse=True)
    n_subj = int(inv.max() + 1)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_subj)
    n_hold = int(round(args.holdout_frac * n_subj))
    hold = np.sort(perm[:n_hold])
    train = np.sort(perm[n_hold:])

    np.save(args.out_dir / "holdout_subj_ids.npy", hold)
    np.save(args.out_dir / "train_subj_ids.npy", train)
    print(f"n_subjects = {n_subj}; holdout = {len(hold)} "
          f"({args.holdout_frac*100:.0f}%), train = {len(train)}")
    # Cohort-level mortality split (informational)
    Y = cohort.Y.numpy()
    at_risk = cohort.at_risk.numpy().squeeze(-1)
    # Subject-level Y: did this subject ever have Y=1 in any at-risk slot?
    y_subj = np.zeros(cohort.Y.shape[0], dtype=int)
    for i in range(cohort.Y.shape[0]):
        valid = at_risk[i] > 0
        y_subj[i] = int(Y[i][valid].max()) if valid.any() else 0
    # group_idx per stay (use subject_ids inverse)
    stay_to_subj = inv  # (n_stays,)
    # Subject-level mortality (any stay had Y=1)
    subj_y = np.zeros(n_subj, dtype=int)
    for stay_idx in range(len(stay_to_subj)):
        subj_y[stay_to_subj[stay_idx]] |= y_subj[stay_idx]
    print(f"  train mortality = {subj_y[train].mean()*100:.2f}%, "
          f"holdout mortality = {subj_y[hold].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
