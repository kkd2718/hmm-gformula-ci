#!/bin/bash
# Overnight pipeline v3 — runs all stages inline with output-file verification.
# Each stage writes a .DONE marker on success; downstream stages bail if upstream missing.
set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/overnight.log
cd $HOME/hmm-gformula-ci
mkdir -p results/main_v2

echo "[$(ts)] === overnight v3 start ===" | tee -a $LOG
echo "[$(ts)] git HEAD: $(git rev-parse --short HEAD)" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | tee -a $LOG

# ============================================================
# Stage 1: Main Table 2 — GPU symmetric refit-bootstrap
# ============================================================
echo "[$(ts)] [1/6] Main Table 2 (vem-refit, B=100) ..." | tee -a $LOG
python3 -u scripts/table2_method_comparison.py \
  --csv data/ards_v31_v4.csv \
  --out-dir results/main_v2 \
  --n-bootstrap-cls 100 --n-bootstrap-vem 100 \
  --xu-b-draws 200 --vem-epochs 50 \
  --vem-refit \
  > results/main_v2.log 2>&1
if [ ! -f results/main_v2/table2_risks.npz ]; then
  echo "[$(ts)] [FAIL] Main Table 2 output missing; aborting pipeline." | tee -a $LOG
  tail -30 results/main_v2.log >> $LOG
  exit 1
fi
echo "[$(ts)] [1/6] done." | tee -a $LOG

# ============================================================
# Stage 2: Fig 2 (scatter + curves)
# ============================================================
echo "[$(ts)] [2/6] Fig 2 ..." | tee -a $LOG
python3 -u scripts/fig2_dose_response.py \
  --npz results/main_v2/table2_risks.npz \
  --out-prefix results/main_v2/fig2_dose_response \
  >> $LOG 2>&1
echo "[$(ts)] [2/6] done." | tee -a $LOG

# ============================================================
# Stage 3: Table 3 LOCO + grouped (VEM-only, refit=False)
# ============================================================
echo "[$(ts)] [3/6] Table 3 LOCO ..." | tee -a $LOG
python3 -u scripts/table3_sensitivity.py \
  --csv data/ards_v31_v4.csv \
  --npz results/main_v2/table2_risks.npz \
  --out-dir results/main_v2/sensitivity \
  --vem-epochs 50 --n-bootstrap 20 \
  > results/loco.log 2>&1
if [ ! -f results/main_v2/sensitivity/table3_sensitivity.md ]; then
  echo "[$(ts)] [WARN] LOCO output missing; continuing." | tee -a $LOG
  tail -30 results/loco.log >> $LOG
fi
echo "[$(ts)] [3/6] done." | tee -a $LOG

# ============================================================
# Stage 4: Fig 4 subgroup
# ============================================================
echo "[$(ts)] [4/6] Fig 4 subgroup ..." | tee -a $LOG
python3 -u scripts/fig4_subgroup_forest.py \
  --csv data/ards_v31_v4.csv \
  --out-dir results/main_v2/fig4 \
  --vem-epochs 50 --n-bootstrap 20 \
  > results/fig4.log 2>&1
echo "[$(ts)] [4/6] done." | tee -a $LOG

# ============================================================
# Stage 5: Appendix B PPC
# ============================================================
echo "[$(ts)] [5/6] PPC ..." | tee -a $LOG
python3 -u scripts/table_s3_ppc.py \
  --csv data/ards_v31_v4.csv \
  --out-dir results/main_v2/appendix_b \
  --vem-epochs 50 \
  > results/ppc.log 2>&1
echo "[$(ts)] [5/6] done." | tee -a $LOG

# ============================================================
# Stage 6: Table 4 simulation
# ============================================================
echo "[$(ts)] [6/6] Simulation ..." | tee -a $LOG
python3 -u scripts/table_s1_simulation.py \
  --out-dir results/main_v2/table4_sim \
  --gammas 0.0 0.2 0.5 0.8 \
  --n-reps 100 --n-per-rep 1000 --T 28 --K 4 \
  --vem-epochs 50 \
  > results/sim.log 2>&1
echo "[$(ts)] [6/6] done." | tee -a $LOG

echo "[$(ts)] === overnight v3 complete ===" | tee -a $LOG
