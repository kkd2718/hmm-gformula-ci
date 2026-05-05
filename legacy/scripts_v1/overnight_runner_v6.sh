#!/bin/bash
# Overnight v6 — re-run all main + appendices with vem-epochs=300 (post-plateau).
# Option B (primary) + Option A (sensitivity) + Natural-course bootstrap + Simulation.
set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/overnight_v6.log
cd $HOME/hmm-gformula-ci
mkdir -p results

VEM_EPOCHS=300

echo "[$(ts)] === overnight v6 start (vem-epochs=$VEM_EPOCHS) ===" | tee -a $LOG
echo "[$(ts)] git HEAD: $(git rev-parse --short HEAD)" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | tee -a $LOG

run_pipeline() {
  local OPTION_TAG=$1
  local Z_LAG_FLAG=$2
  local OUT=$HOME/hmm-gformula-ci/results/main_v6_option${OPTION_TAG}

  echo "[$(ts)] ===== Option ${OPTION_TAG} pipeline ${OUT} =====" | tee -a $LOG
  mkdir -p $OUT

  # 1) Main Table 2
  echo "[$(ts)] [${OPTION_TAG}/1/5] Main Table 2 ..." | tee -a $LOG
  python3 -u scripts/table2_method_comparison.py \
    --csv data/ards_v31_v4.csv --out-dir $OUT \
    --n-bootstrap-cls 100 --n-bootstrap-vem 100 \
    --xu-b-draws 200 --vem-epochs $VEM_EPOCHS \
    --vem-refit ${Z_LAG_FLAG} \
    > $OUT/main.log 2>&1
  if [ ! -f $OUT/table2_risks.npz ]; then
    echo "[$(ts)] [${OPTION_TAG}/1/5] FAIL — Main missing; skip rest" | tee -a $LOG
    tail -30 $OUT/main.log >> $LOG; return 1
  fi
  echo "[$(ts)] [${OPTION_TAG}/1/5] done." | tee -a $LOG

  # 2) Fig 2
  echo "[$(ts)] [${OPTION_TAG}/2/5] Fig 2 ..." | tee -a $LOG
  python3 -u scripts/fig2_dose_response.py \
    --npz $OUT/table2_risks.npz --out-prefix $OUT/fig2_dose_response \
    >> $LOG 2>&1
  echo "[$(ts)] [${OPTION_TAG}/2/5] done." | tee -a $LOG

  # 3) LOCO
  echo "[$(ts)] [${OPTION_TAG}/3/5] LOCO ..." | tee -a $LOG
  python3 -u scripts/table3_sensitivity.py \
    --csv data/ards_v31_v4.csv --npz $OUT/table2_risks.npz \
    --out-dir $OUT/sensitivity --vem-epochs $VEM_EPOCHS --n-bootstrap 20 ${Z_LAG_FLAG} \
    > $OUT/loco.log 2>&1
  echo "[$(ts)] [${OPTION_TAG}/3/5] done." | tee -a $LOG

  # 4) Fig 4
  echo "[$(ts)] [${OPTION_TAG}/4/5] Fig 4 ..." | tee -a $LOG
  python3 -u scripts/fig4_subgroup_forest.py \
    --csv data/ards_v31_v4.csv --out-dir $OUT/fig4 \
    --vem-epochs $VEM_EPOCHS --n-bootstrap 20 ${Z_LAG_FLAG} \
    > $OUT/fig4.log 2>&1
  echo "[$(ts)] [${OPTION_TAG}/4/5] done." | tee -a $LOG

  # 5) PPC
  echo "[$(ts)] [${OPTION_TAG}/5/5] PPC ..." | tee -a $LOG
  python3 -u scripts/table_s3_ppc.py \
    --csv data/ards_v31_v4.csv --out-dir $OUT/appendix_b \
    --vem-epochs $VEM_EPOCHS ${Z_LAG_FLAG} \
    > $OUT/ppc.log 2>&1
  echo "[$(ts)] [${OPTION_TAG}/5/5] done." | tee -a $LOG
}

run_pipeline "B" "--vem-z-lag-treatment"
run_pipeline "A" ""

# Natural-course bootstrap (B=100, Option B for primary)
echo "[$(ts)] [F] Natural-course bootstrap (B=100) ..." | tee -a $LOG
python3 -u scripts/natural_course_bootstrap.py \
  --csv data/ards_v31_v4.csv --out-dir results/appendix_f_v6 \
  --n-bootstrap 100 --vem-epochs $VEM_EPOCHS --vem-z-lag-treatment \
  > results/appendix_f_v6.log 2>&1
echo "[$(ts)] [F] done." | tee -a $LOG

# Simulation refactor (Appendix E)
echo "[$(ts)] [E] Simulation refactor ..." | tee -a $LOG
python3 -u scripts/table_s1_simulation.py \
  --out-dir results/appendix_e_v6 \
  --gammas 0.0 0.2 0.5 0.8 \
  --n-reps 100 --n-per-rep 5000 --T 28 --K 4 \
  --vem-epochs $VEM_EPOCHS \
  > results/appendix_e_v6.log 2>&1
echo "[$(ts)] [E] done." | tee -a $LOG

echo "[$(ts)] === overnight v6 complete ===" | tee -a $LOG
