#!/bin/bash
# Overnight v5 — appendix E (refactored simulation) + appendix F (natural-course bootstrap).
set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/overnight_v5.log
cd $HOME/hmm-gformula-ci
mkdir -p results

echo "[$(ts)] === overnight v5 start ===" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | tee -a $LOG

# 1) Natural-course bootstrap (Appendix F) — B=100, 3 methods
echo "[$(ts)] [1/2] Natural-course bootstrap (B=100, 3 methods, Option B) ..." | tee -a $LOG
python3 -u scripts/natural_course_bootstrap.py \
  --csv data/ards_v31_v4.csv \
  --out-dir results/appendix_f \
  --n-bootstrap 100 \
  --vem-epochs 50 \
  --vem-z-lag-treatment \
  > results/appendix_f.log 2>&1
if [ -f results/appendix_f/appendix_f_natural_course.md ]; then
  echo "[$(ts)] [1/2] done." | tee -a $LOG
else
  echo "[$(ts)] [1/2] FAIL — output missing" | tee -a $LOG
  tail -30 results/appendix_f.log >> $LOG
fi

# 2) Refactored simulation (Appendix E) — bias/RMSE focus, 4 gammas, larger N
echo "[$(ts)] [2/2] Simulation refactor (n_reps=100, N=5000, vem-epochs=100) ..." | tee -a $LOG
python3 -u scripts/table_s1_simulation.py \
  --out-dir results/appendix_e \
  --gammas 0.0 0.2 0.5 0.8 \
  --n-reps 100 --n-per-rep 5000 --T 28 --K 4 \
  --vem-epochs 100 \
  > results/appendix_e.log 2>&1
echo "[$(ts)] [2/2] done." | tee -a $LOG

echo "[$(ts)] === overnight v5 complete ===" | tee -a $LOG
