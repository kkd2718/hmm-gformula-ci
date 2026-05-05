#!/bin/bash
# Round-2 chain Stage 2-6 (run after K=1 LOCO completes)
# Stage 2: K=1 + K=5 NUTS refit with --record-loglik (~25 min total)
# Stage 3: WAIC + PSIS-LOO computation (~5 min, no GPU)
# Stage 4: Spec II re-fit recording lambda_L explicitly (~10 min)
# Stage 5: Prior sensitivity K=5 (Gamma) (~10 min fit + 1 min dose)
# Stage 6: Holdout split + K=5 PPC fit (80%) + posterior-predict 20% (~15 min)

set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/round2_chain_b.log
cd $HOME/hmm-gformula-ci

echo "[$(ts)] === Round-2 chain B start ===" | tee -a $LOG

# Wait for any running run_bayesian or run_jax processes
echo "[$(ts)] Waiting for Stage 1 (K=1 LOCO) ..." | tee -a $LOG
while pgrep -f "round2_chain.sh" > /dev/null \
   || pgrep -f "run_bayesian_main.py.*loco_K1" > /dev/null; do
    sleep 60
done
echo "[$(ts)] Stage 1 finished, starting Stage 2." | tee -a $LOG

# ----- Stage 2: NUTS refits with log_lik recording -----
mkdir -p results/loglik_main
echo "[$(ts)] === Stage 2: K=1 + K=5 NUTS with log_lik ===" | tee -a $LOG

python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/loglik_main \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K1 --phase fit --record-loglik \
  >> $LOG 2>&1

python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/loglik_main \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K5 --phase fit --record-loglik \
  >> $LOG 2>&1

# Also Xu Bayesian for completeness (optional but cheap for diagnostics)
python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/loglik_main \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods xu --phase fit --record-loglik \
  >> $LOG 2>&1

# ----- Stage 3: WAIC + LOO -----
echo "[$(ts)] === Stage 3: WAIC + PSIS-LOO ===" | tee -a $LOG
python3 -u scripts/compute_waic_loo.py \
  --state-files results/loglik_main/fre_nice_K1_state.npz \
                results/loglik_main/fre_nice_K5_state.npz \
                results/loglik_main/xu_bayesian_state.npz \
  --labels "K=1 NICE" "K=5 FRE-NICE" "Xu Bayesian" \
  --out-md results/loglik_main/waic_loo_table.md \
  >> $LOG 2>&1

# Diagnostics summary across methods
echo "[$(ts)] === Stage 3b: convergence diagnostics ===" | tee -a $LOG
python3 -u scripts/extract_diagnostics.py \
  --state-files results/loglik_main/fre_nice_K1_state.npz \
                results/loglik_main/fre_nice_K5_state.npz \
                results/loglik_main/xu_bayesian_state.npz \
                results/bayesian_spec2/fre_nice_K5_state.npz \
  --labels "K=1 NICE" "K=5 FRE-NICE" "Xu Bayesian" "K=5 Spec II" \
  --out-md results/loglik_main/diagnostics_table.md \
  >> $LOG 2>&1

# ----- Stage 4: Spec II re-fit with explicit lambda_L (and log_lik for completeness) -----
mkdir -p results/spec2_v3
echo "[$(ts)] === Stage 4: Spec II re-fit with explicit lambda_L ===" | tee -a $LOG
python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/spec2_v3 \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K5 --phase fit --share-RE-on-L --record-loglik \
  >> $LOG 2>&1

# ----- Stage 5: Prior sensitivity (Gamma on sigma_b) -----
mkdir -p results/prior_sens_gamma
echo "[$(ts)] === Stage 5: Prior sensitivity K=5 (Gamma) ===" | tee -a $LOG
python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/prior_sens_gamma \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K5 --phase fit --sigma-b-prior gamma \
  >> $LOG 2>&1
python3 -u scripts/run_jax_dose.py \
  --csv data/ards_v31_v4.csv --state-dir results/prior_sens_gamma \
  --out-dir results/prior_sens_gamma --prefix fre_nice_K5 \
  >> $LOG 2>&1

# ----- Stage 6: Holdout PPC -----
mkdir -p results/ppc_K5
echo "[$(ts)] === Stage 6a: holdout split ===" | tee -a $LOG
python3 -u scripts/make_holdout_split.py \
  --csv data/ards_v31_v4.csv --holdout-frac 0.2 --seed 42 \
  --out-dir results/ppc_K5 >> $LOG 2>&1

echo "[$(ts)] === Stage 6b: K=5 fit on 80% (holdout masked) ===" | tee -a $LOG
python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/ppc_K5 \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K5 --phase fit \
  --holdout-subj-ids-file results/ppc_K5/holdout_subj_ids.npy \
  >> $LOG 2>&1

echo "[$(ts)] === Stage 6c: posterior predict held-out ===" | tee -a $LOG
python3 -u scripts/ppc_holdout.py \
  --csv data/ards_v31_v4.csv \
  --state-path results/ppc_K5/fre_nice_K5_state.npz \
  --holdout-ids results/ppc_K5/holdout_subj_ids.npy \
  --out-md results/ppc_K5/ppc_K5.md \
  --n-posterior-subset 200 --n-b-draws 5 \
  >> $LOG 2>&1

echo "[$(ts)] === Round-2 chain B complete ===" | tee -a $LOG
