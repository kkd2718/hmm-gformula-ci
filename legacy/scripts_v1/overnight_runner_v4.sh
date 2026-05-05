#!/bin/bash
# Overnight v4 — runs Option A and Option B sequentially, full pipeline each.
# Generates main_v3_optionA/ and main_v3_optionB/ result trees.
set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/overnight_v4.log
cd $HOME/hmm-gformula-ci
mkdir -p results

echo "[$(ts)] === overnight v4 start (Option A + B chained) ===" | tee -a $LOG
echo "[$(ts)] git HEAD: $(git rev-parse --short HEAD)" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | tee -a $LOG

run_pipeline() {
  local OPTION_TAG=$1     # "A" or "B"
  local Z_LAG_FLAG=$2     # "" or "--vem-z-lag-treatment"
  local OUT=$HOME/hmm-gformula-ci/results/main_v3_option${OPTION_TAG}

  echo "[$(ts)] ===== Option ${OPTION_TAG} pipeline ${OUT} =====" | tee -a $LOG
  mkdir -p $OUT

  # 1) Main Table 2 (vem-refit, B=100)
  echo "[$(ts)] [1/5] Main Table 2 ..." | tee -a $LOG
  python3 -u scripts/table2_method_comparison.py \
    --csv data/ards_v31_v4.csv \
    --out-dir $OUT \
    --n-bootstrap-cls 100 --n-bootstrap-vem 100 \
    --xu-b-draws 200 --vem-epochs 50 \
    --vem-refit ${Z_LAG_FLAG} \
    > $OUT/main_v3.log 2>&1
  if [ ! -f $OUT/table2_risks.npz ]; then
    echo "[$(ts)] [FAIL ${OPTION_TAG}] Main Table 2 missing; skipping rest of ${OPTION_TAG}." | tee -a $LOG
    tail -30 $OUT/main_v3.log >> $LOG
    return 1
  fi
  echo "[$(ts)] [1/5] done." | tee -a $LOG

  # 2) Fig 2
  echo "[$(ts)] [2/5] Fig 2 ..." | tee -a $LOG
  python3 -u scripts/fig2_dose_response.py \
    --npz $OUT/table2_risks.npz \
    --out-prefix $OUT/fig2_dose_response \
    >> $LOG 2>&1
  echo "[$(ts)] [2/5] done." | tee -a $LOG

  # 3) Table 3 LOCO
  echo "[$(ts)] [3/5] LOCO ..." | tee -a $LOG
  python3 -u scripts/table3_sensitivity.py \
    --csv data/ards_v31_v4.csv \
    --npz $OUT/table2_risks.npz \
    --out-dir $OUT/sensitivity \
    --vem-epochs 50 --n-bootstrap 20 ${Z_LAG_FLAG} \
    > $OUT/loco.log 2>&1
  echo "[$(ts)] [3/5] done." | tee -a $LOG

  # 4) Fig 4 subgroup
  echo "[$(ts)] [4/5] Fig 4 subgroup ..." | tee -a $LOG
  python3 -u scripts/fig4_subgroup_forest.py \
    --csv data/ards_v31_v4.csv \
    --out-dir $OUT/fig4 \
    --vem-epochs 50 --n-bootstrap 20 ${Z_LAG_FLAG} \
    > $OUT/fig4.log 2>&1
  echo "[$(ts)] [4/5] done." | tee -a $LOG

  # 5) PPC
  echo "[$(ts)] [5/5] PPC ..." | tee -a $LOG
  python3 -u scripts/table_s3_ppc.py \
    --csv data/ards_v31_v4.csv \
    --out-dir $OUT/appendix_b \
    --vem-epochs 50 ${Z_LAG_FLAG} \
    > $OUT/ppc.log 2>&1
  echo "[$(ts)] [5/5] done." | tee -a $LOG
}

# Option B first (primary spec per Obsidian/study_design.md)
run_pipeline "B" "--vem-z-lag-treatment"
# Option A as conservative sensitivity
run_pipeline "A" ""

# 6) Simulation (single run with Option B since simulation DGP uses it as ground)
echo "[$(ts)] [6] Simulation ..." | tee -a $LOG
python3 -u scripts/table_s1_simulation.py \
  --out-dir $HOME/hmm-gformula-ci/results/main_v3_simulation \
  --gammas 0.0 0.2 0.5 0.8 \
  --n-reps 100 --n-per-rep 1000 --T 28 --K 4 \
  --vem-epochs 50 \
  > $HOME/hmm-gformula-ci/results/sim.log 2>&1
echo "[$(ts)] [6] done." | tee -a $LOG

echo "[$(ts)] === overnight v4 complete ===" | tee -a $LOG
