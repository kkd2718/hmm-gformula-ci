#!/bin/bash
# Overnight runner: waits for Main Table 2 to finish, then runs queued tasks.
set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/overnight.log
cd $HOME/hmm-gformula-ci

echo "[$(ts)] === post-main runner v2 start ===" | tee -a $LOG

# 1) Wait for Main Table 2 (already running)
echo "[$(ts)] waiting for Main Table 2 ..." | tee -a $LOG
while pgrep -f table2_method_comparison > /dev/null; do sleep 60; done
echo "[$(ts)] Main Table 2 done." | tee -a $LOG

# 2) Fig 2 (scatter + curves) — instant
echo "[$(ts)] [Fig 2] scatter + curves ..." | tee -a $LOG
python3 -u scripts/fig2_dose_response.py \
  --npz results/main_v2/table2_risks.npz \
  --out-prefix results/main_v2/fig2_dose_response \
  >> $LOG 2>&1

# 3) Table 3 LOCO + grouped (VEM-only, refit=False)
echo "[$(ts)] [LOCO] starting ..." | tee -a $LOG
python3 -u scripts/table3_sensitivity.py \
  --csv data/ards_v31_v4.csv \
  --npz results/main_v2/table2_risks.npz \
  --out-dir results/main_v2/sensitivity \
  --vem-epochs 50 --n-bootstrap 20 \
  > results/loco.log 2>&1
echo "[$(ts)] [LOCO] done." | tee -a $LOG

# 4) Fig 4 subgroup forest (VEM-only)
echo "[$(ts)] [Fig 4] starting ..." | tee -a $LOG
python3 -u scripts/fig4_subgroup_forest.py \
  --csv data/ards_v31_v4.csv \
  --out-dir results/main_v2/fig4 \
  --vem-epochs 50 --n-bootstrap 20 \
  > results/fig4.log 2>&1
echo "[$(ts)] [Fig 4] done." | tee -a $LOG

# 5) Appendix B PPC
echo "[$(ts)] [PPC] starting ..." | tee -a $LOG
python3 -u scripts/table_s3_ppc.py \
  --csv data/ards_v31_v4.csv \
  --out-dir results/main_v2/appendix_b \
  --vem-epochs 50 \
  > results/ppc.log 2>&1
echo "[$(ts)] [PPC] done." | tee -a $LOG

# 6) Table 4 simulation
echo "[$(ts)] [Simulation] starting ..." | tee -a $LOG
python3 -u scripts/table_s1_simulation.py \
  --out-dir results/main_v2/table4_sim \
  --gammas 0.0 0.2 0.5 0.8 \
  --n-reps 100 --n-per-rep 1000 --T 28 --K 4 \
  --vem-epochs 50 \
  > results/sim.log 2>&1
echo "[$(ts)] [Simulation] done." | tee -a $LOG

echo "[$(ts)] === post-main runner v2 complete ===" | tee -a $LOG
