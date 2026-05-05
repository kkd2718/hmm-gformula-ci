#!/bin/bash
# Parallel pipeline for v2 (ref-bin-fixed) 4-method run on V100.
#
# Pipelining: overlap dose_response (CPU) with next method's NUTS fit (GPU).
# Stages:
#   A: Standard NICE (CPU)         + Xu_fit (GPU)    [parallel]
#   B: Xu_dose (CPU)               + K1_fit (GPU)    [parallel]
#   C: K1_dose (CPU)               + K5_fit (GPU)    [parallel]
#   D: K5_dose (CPU)
#
# Total time ~3-3.5h (vs 7h sequential).
set -e
cd ~/hmm-gformula-ci

OUT_DIR=results/bayesian_main_v2
STD_DIR=results/standard_v2
LOG=results/run_v2_pipeline.log
mkdir -p $OUT_DIR $STD_DIR

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

echo "[$(ts)] === Pipeline v2 start ===" | tee -a $LOG

# Common Bayesian args
BAYES="--csv data/ards_v31_v4.csv --out-dir $OUT_DIR --inference nuts \
  --n-warmup 1000 --n-samples 1000 --n-chains 2 --target-accept 0.95 \
  --n-b-draws 50 --n-posterior-subset 200"

# ----- Stage A: Standard NICE (background CPU) + Xu_fit (foreground GPU) -----
echo "[$(ts)] [A] Launching Standard NICE (CPU) in background ..." | tee -a $LOG
nohup python3 -u scripts/table2_method_comparison.py \
  --csv data/ards_v31_v4.csv --out-dir $STD_DIR \
  --skip xu vem --n-bootstrap-cls 100 --reference-mp 17.0 \
  > $STD_DIR/standard.log 2>&1 &
STD_PID=$!
echo "[$(ts)] [A] Standard PID=$STD_PID" | tee -a $LOG

echo "[$(ts)] [A] Xu Bayesian FIT (GPU) ..." | tee -a $LOG
python3 -u scripts/run_bayesian_main.py $BAYES --methods xu --phase fit \
  >> $LOG 2>&1
echo "[$(ts)] [A] Xu fit done." | tee -a $LOG

# ----- Stage B: Xu_dose (background CPU) + K1_fit (foreground GPU) -----
echo "[$(ts)] [B] Xu DOSE (CPU) in background ..." | tee -a $LOG
nohup python3 -u scripts/run_bayesian_main.py $BAYES --methods xu --phase dose \
  > $OUT_DIR/xu_dose.log 2>&1 &
XU_DOSE_PID=$!
echo "[$(ts)] [B] Xu_dose PID=$XU_DOSE_PID" | tee -a $LOG

echo "[$(ts)] [B] K=1 FRE-NICE FIT (GPU) ..." | tee -a $LOG
python3 -u scripts/run_bayesian_main.py $BAYES --methods K1 --phase fit \
  >> $LOG 2>&1
echo "[$(ts)] [B] K=1 fit done." | tee -a $LOG

# ----- Stage C: K1_dose (background CPU) + K5_fit (foreground GPU) -----
echo "[$(ts)] [C] K=1 DOSE (CPU) in background ..." | tee -a $LOG
nohup python3 -u scripts/run_bayesian_main.py $BAYES --methods K1 --phase dose \
  > $OUT_DIR/K1_dose.log 2>&1 &
K1_DOSE_PID=$!
echo "[$(ts)] [C] K1_dose PID=$K1_DOSE_PID" | tee -a $LOG

echo "[$(ts)] [C] K=5 FRE-NICE FIT (GPU) ..." | tee -a $LOG
python3 -u scripts/run_bayesian_main.py $BAYES --methods K5 --phase fit \
  >> $LOG 2>&1
echo "[$(ts)] [C] K=5 fit done." | tee -a $LOG

# ----- Stage D: K5_dose -----
echo "[$(ts)] [D] K=5 DOSE (CPU) ..." | tee -a $LOG
python3 -u scripts/run_bayesian_main.py $BAYES --methods K5 --phase dose \
  >> $LOG 2>&1

# Wait for any lingering background dose processes
echo "[$(ts)] Waiting for background dose processes (Standard, Xu_dose, K1_dose) ..." | tee -a $LOG
wait $STD_PID $XU_DOSE_PID $K1_DOSE_PID 2>/dev/null
echo "[$(ts)] All background processes complete." | tee -a $LOG

# Generate combined Bayesian table (re-run dose phase 'full' just for table — fast)
echo "[$(ts)] Generating combined Bayesian table2.md ..." | tee -a $LOG
python3 -u scripts/run_bayesian_main.py $BAYES --methods xu K1 K5 --phase dose \
  >> $LOG 2>&1
echo "[$(ts)] === Pipeline v2 complete ===" | tee -a $LOG
