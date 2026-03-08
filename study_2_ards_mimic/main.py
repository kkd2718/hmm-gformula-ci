"""
study_2_ards_mimic/main.py (v4.1)
====================================================
Title: Dynamic Causal Inference of Mechanical Power in Severe ARDS
       — A Latent State Modeling Approach —

=============================================================
실행 파이프라인 (반드시 이 순서대로):
  1. python preprocessing/mimic-extract.py   → CSV 생성 (Gattinoni MP)
  2. python preprocessing/hmm-processing.py  → PyTorch 텐서 생성
  3. python main.py                          → 전체 실험 + 결과 저장

  ※ mimic-extract.py에서 MP 공식이 변경되었으므로
    hmm-processing.py도 반드시 재실행해야 합니다.
    (텐서가 CSV를 읽어서 생성하기 때문)
=============================================================

비교 모델 (3개, 각각 명확한 존재 이유):
  1. Proposed Model (ContinuousHMM)
     - Latent state-space + binned MP + g-formula
     - 핵심 기여: time-varying confounding 교정 + 비선형 dose-response
  2. Standard G-formula (StandardGFormula)
     - 동일 구조에서 Z만 제거 → ablation study
     - 보여주는 것: latent state의 기여도
  3. Cox PH (CoxPHBaseline, lifelines)
     - 임상 표준 분석 → clinical benchmark
     - 보여주는 것: time-varying confounding 미보정의 결과
"""

import os, sys, torch, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

# Configuration
torch.set_num_threads(os.cpu_count())
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'figure.dpi': 300,
    'axes.labelsize': 12, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 9,
})

# =============================================================================
# Hyperparameters
# =============================================================================
N_BINS = 20
LAMBDA_SMOOTH = 0.02
N_EPOCHS = 300
N_EPOCHS_BOOT = 100
N_BOOTSTRAPS = 100
LR = 0.01

# =============================================================================
# Paths
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.dynamic_hmm import ContinuousHMM, StandardGFormula, CoxPHBaseline

DATA_TENSOR = os.path.join(CURRENT_DIR, 'processed_data', 'hmm_tensor', 'mimic_hmm_tensor.pt')
DATA_CSV = os.path.join(CURRENT_DIR, 'processed_data', 'final_cohort', 'ards_standard_cohort.csv')
SCALER_FILE = os.path.join(CURRENT_DIR, 'processed_data', 'hmm_tensor', 'scalers.pkl')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'study_2_ards_mimic', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def plot_smooth_ci(ax, x, mean, lower, upper, color, label,
                   linestyle='-', alpha=0.15):
    x_new = np.linspace(x.min(), x.max(), 300)
    try:
        m_s = np.clip(make_interp_spline(x, mean, k=3)(x_new), 0, 100)
        l_s = np.clip(make_interp_spline(x, lower, k=3)(x_new), 0, 100)
        u_s = np.clip(make_interp_spline(x, upper, k=3)(x_new), 0, 100)
        ax.plot(x_new, m_s, linestyle=linestyle, color=color,
                label=label, linewidth=2)
        ax.fill_between(x_new, l_s, u_s, color=color, alpha=alpha, edgecolor=None)
    except Exception:
        ax.plot(x, mean, linestyle=linestyle, color=color,
                label=label, linewidth=2)
        ax.fill_between(x, lower, upper, color=color, alpha=alpha)


def standardize_mp(mp_jmin, scaler_S):
    return (np.log(mp_jmin) - scaler_S.mean_[0]) / scaler_S.scale_[0]


# =============================================================================
# Main
# =============================================================================
def run_all():
    print("=" * 60)
    print("  Study 2: Dynamic Causal Inference of MP in Severe ARDS")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 0. Load Data
    # -----------------------------------------------------------------
    print("\n📂 Loading data...")
    data = torch.load(DATA_TENSOR, weights_only=False)
    sc = joblib.load(SCALER_FILE)

    S = data['S']           # (N, T, 1)
    L_dyn = data['L']       # (N, T, 3)
    C_static = data['C']    # (N, 4)
    Y = data['Y']           # (N, T, 1)
    mask = data['mask']     # (N, T)
    sc_S = sc['S']

    df = pd.read_csv(DATA_CSV)
    base_pf = df[df['day_num'] == 0].groupby('stay_id')['pf_ratio'].first().values

    N_patients = S.shape[0]
    if len(base_pf) > N_patients:
        base_pf = base_pf[:N_patients]

    print(f"   Patients: {N_patients}, Time steps: {S.shape[1]}")
    print(f"   Dynamic covariates: {L_dyn.shape[2]}, Static: {C_static.shape[1]}")

    n_dyn = L_dyn.shape[2]
    n_static = C_static.shape[1]

    # -----------------------------------------------------------------
    # 1. Table 1 & Figure 1
    # -----------------------------------------------------------------
    print("\n📊 Generating Table 1 & Figure 1...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    base_df = df.groupby('stay_id').first().reset_index()
    t1_vars = {
        'anchor_age': 'Age', 'charlson_index': 'CCI',
        'pf_ratio': 'P/F Ratio', 'lactate': 'Lactate',
        'compliance': 'Compliance', 'mp_j_min': 'MP (J/min)',
    }
    t1_data = []
    for v, label in t1_vars.items():
        s = base_df[base_df['hospital_expire_flag'] == 0][v]
        d = base_df[base_df['hospital_expire_flag'] == 1][v]
        t1_data.append({
            'Variable': label,
            'Survivors': f"{s.mean():.1f} ({s.std():.1f})",
            'Non-Survivors': f"{d.mean():.1f} ({d.std():.1f})",
        })
    pd.DataFrame(t1_data).to_csv(
        os.path.join(OUTPUT_DIR, 'Table1_Baseline.csv'), index=False
    )

    # Forest plot
    base_f = df[df['day_num'] == 0].dropna(subset=['mp_j_min', 'pf_ratio']).copy()
    base_f['mp_cat'] = pd.cut(
        base_f['mp_j_min'], bins=[0, 12, 17, 22, 100],
        labels=['<12', '12-17', '17-22', '>22']
    )
    X_num = StandardScaler().fit_transform(
        base_f[['anchor_age', 'charlson_index', 'pf_ratio',
                'lactate', 'compliance']].fillna(0)
    )
    X_cat = base_f[['is_male']].values
    X_mp = pd.get_dummies(
        base_f['mp_cat'], prefix='MP', dtype=float
    )[['MP_<12', 'MP_17-22', 'MP_>22']].values
    X = np.hstack([X_num, X_cat, X_mp])
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    y = base_f['hospital_expire_flag'].values

    clf = LogisticRegression(penalty=None, fit_intercept=False, max_iter=5000)
    clf.fit(X_design, y)
    probs = clf.predict_proba(X_design)[:, 1]
    W = probs * (1 - probs)
    Hessian = X_design.T @ (X_design * W[:, np.newaxis])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(Hessian)))
    except np.linalg.LinAlgError:
        se = np.full(len(clf.coef_[0]), 0.1)

    feat_names = [
        'Intercept', 'Age', 'CCI', 'P/F Ratio', 'Lactate',
        'Compliance', 'Sex (Male)', 'MP < 12', 'MP 17-22', 'MP > 22',
    ]
    or_df = pd.DataFrame({
        'Feature': feat_names, 'OR': np.exp(clf.coef_[0]), 'SE': se,
        'Lower': np.exp(clf.coef_[0] - 1.96 * se),
        'Upper': np.exp(clf.coef_[0] + 1.96 * se),
    })
    or_df = or_df[or_df['Feature'] != 'Intercept']
    or_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_Fig1_Forest.csv'), index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(
        or_df['OR'], range(len(or_df)),
        xerr=[or_df['OR'] - or_df['Lower'], or_df['Upper'] - or_df['OR']],
        fmt='s', color='black', capsize=5,
    )
    ax.axvline(1, linestyle='--', color='red')
    ax.set_yticks(range(len(or_df)))
    ax.set_yticklabels(or_df['Feature'])
    ax.set_xlabel('Adjusted Odds Ratio (95% CI)')
    ax.set_title('Figure 1. Baseline Association')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'Figure1_Forest_Plot.png'))
    plt.close(fig)

    # -----------------------------------------------------------------
    # 2. Model Training & G-Formula Simulation
    # -----------------------------------------------------------------
    print("\n🧠 Training Models...")

    idx_sev = np.where(base_pf <= 100)[0]
    idx_mod = np.where((base_pf > 100) & (base_pf <= 200))[0]
    idx_mild = np.where(base_pf > 200)[0]
    indices = {'Severe': idx_sev, 'Moderate': idx_mod, 'Mild': idx_mild}

    targets = np.linspace(5, 35, 61)

    res_strat = []
    comp_results = []
    cox_full = None
    model_full = None

    for grp, idx in indices.items():
        if len(idx) < 50:
            print(f"   Skipping {grp}: n={len(idx)} < 50")
            continue

        print(f"\n--- {grp} ARDS (n={len(idx)}) ---")

        S_g, L_g, C_g = S[idx], L_dyn[idx], C_static[idx]
        Y_g, M_g = Y[idx], mask[idx]

        # === Full-data fit ===
        print("   Fitting proposed model (Latent+Smooth)...")
        model_full = ContinuousHMM(n_dyn, n_static, N_BINS)
        model_full.fit(S_g, L_g, C_g, Y_g, M_g,
                       n_epochs=N_EPOCHS, lambda_smooth=LAMBDA_SMOOTH, lr=LR)

        # --- Severe: 2×2 factorial + Cox (5 models total) ---
        #
        # Figure 2A (Clinical): Cox vs Traditional vs Proposed
        # Figure 2B (Ablation): 2×2 factorial
        #
        #                    No Smooth (λ=0)       Smooth (λ=LAMBDA)
        #   No Latent (StdGF)   Traditional          StdGF+Smooth
        #   Latent (HMM)        HMM no-smooth        Proposed
        #
        hmm_nosmooth_full = None
        stdgf_smooth_full = None
        stdgf_nosmooth_full = None
        cox_full_local = None

        if grp == 'Severe':
            print("   Fitting HMM w/o smoothness (Latent+NoSmooth)...")
            hmm_nosmooth_full = ContinuousHMM(n_dyn, n_static, N_BINS)
            hmm_nosmooth_full.fit(S_g, L_g, C_g, Y_g, M_g,
                                  n_epochs=N_EPOCHS, lambda_smooth=0.0, lr=LR)

            print("   Fitting Std G-formula + smooth (NoLatent+Smooth)...")
            stdgf_smooth_full = StandardGFormula(n_dyn, n_static, N_BINS)
            stdgf_smooth_full.fit(S_g, L_g, C_g, Y_g, M_g,
                                  n_epochs=N_EPOCHS, lambda_smooth=LAMBDA_SMOOTH)

            print("   Fitting Traditional G-formula (NoLatent+NoSmooth)...")
            stdgf_nosmooth_full = StandardGFormula(n_dyn, n_static, N_BINS)
            stdgf_nosmooth_full.fit(S_g, L_g, C_g, Y_g, M_g,
                                    n_epochs=N_EPOCHS, lambda_smooth=0.0)

            print("   Fitting Cox PH...")
            cox_full_local = CoxPHBaseline(penalizer=0.01)
            cox_full_local.fit(S_g, L_g, C_g, Y_g, M_g)
            cox_full = cox_full_local

        # === Bootstrap (model refit → parameter uncertainty) ===
        print(f"   Bootstrap ({N_BOOTSTRAPS}×)...")

        # Proposed (Latent+Smooth) — all groups
        boot_hmm = {v: [] for v in targets}
        # Severe-only: 2×2 factorial + Cox
        boot_hmm_nosmooth = {v: [] for v in targets}
        boot_stdgf_smooth = {v: [] for v in targets}
        boot_stdgf_nosmooth = {v: [] for v in targets}
        boot_cox = {v: [] for v in targets}

        for b in tqdm(range(N_BOOTSTRAPS), desc=f"Boot {grp}"):
            ib = np.random.choice(len(idx), len(idx), replace=True)
            S_b, L_b, C_b = S_g[ib], L_g[ib], C_g[ib]
            Y_b, M_b = Y_g[ib], M_g[ib]

            # Proposed (all groups)
            hmm_b = ContinuousHMM(n_dyn, n_static, N_BINS)
            hmm_b.fit(S_b, L_b, C_b, Y_b, M_b,
                      n_epochs=N_EPOCHS_BOOT, lambda_smooth=LAMBDA_SMOOTH, lr=LR)

            for val in targets:
                t_s = standardize_mp(val, sc_S)
                boot_hmm[val].append(
                    hmm_b.simulate_gformula(L_b[:, 0, :], C_b, t_s)[-1] * 100
                )

            if grp == 'Severe':
                # HMM no-smooth
                ns_b = ContinuousHMM(n_dyn, n_static, N_BINS)
                ns_b.fit(S_b, L_b, C_b, Y_b, M_b,
                         n_epochs=N_EPOCHS_BOOT, lambda_smooth=0.0, lr=LR)

                # StdGF + smooth
                sgs_b = StandardGFormula(n_dyn, n_static, N_BINS)
                sgs_b.fit(S_b, L_b, C_b, Y_b, M_b,
                          n_epochs=N_EPOCHS_BOOT, lambda_smooth=LAMBDA_SMOOTH)

                # Traditional (StdGF no-smooth)
                sgns_b = StandardGFormula(n_dyn, n_static, N_BINS)
                sgns_b.fit(S_b, L_b, C_b, Y_b, M_b,
                           n_epochs=N_EPOCHS_BOOT, lambda_smooth=0.0)

                # Cox PH
                cox_b = CoxPHBaseline(penalizer=0.01)
                cox_b.fit(S_b, L_b, C_b, Y_b, M_b)

                for val in targets:
                    t_s = standardize_mp(val, sc_S)
                    boot_hmm_nosmooth[val].append(
                        ns_b.simulate_gformula(L_b[:, 0, :], C_b, t_s)[-1] * 100)
                    boot_stdgf_smooth[val].append(
                        sgs_b.simulate_gformula(L_b[:, 0, :], C_b, t_s)[-1] * 100)
                    boot_stdgf_nosmooth[val].append(
                        sgns_b.simulate_gformula(L_b[:, 0, :], C_b, t_s)[-1] * 100)
                    boot_cox[val].append(
                        cox_b.simulate_gformula(L_b[:, 0, :], C_b, t_s)[-1] * 100)

        # === Collect results ===
        def pct(arr, p):
            return np.percentile(arr, p) if arr else 0.0

        for val in targets:
            res_strat.append({
                'Group': grp, 'MP': val,
                'Mean': np.mean(boot_hmm[val]),
                'Low': pct(boot_hmm[val], 2.5),
                'High': pct(boot_hmm[val], 97.5),
            })
            if grp == 'Severe':
                comp_results.append({
                    'MP': val,
                    # Proposed (Latent + Smooth)
                    'HMM_Mean': np.mean(boot_hmm[val]),
                    'HMM_Low': pct(boot_hmm[val], 2.5),
                    'HMM_High': pct(boot_hmm[val], 97.5),
                    # HMM no-smooth (Latent + NoSmooth)
                    'HMM_NS_Mean': np.mean(boot_hmm_nosmooth[val]),
                    'HMM_NS_Low': pct(boot_hmm_nosmooth[val], 2.5),
                    'HMM_NS_High': pct(boot_hmm_nosmooth[val], 97.5),
                    # StdGF + smooth (NoLatent + Smooth)
                    'SGF_S_Mean': np.mean(boot_stdgf_smooth[val]),
                    'SGF_S_Low': pct(boot_stdgf_smooth[val], 2.5),
                    'SGF_S_High': pct(boot_stdgf_smooth[val], 97.5),
                    # Traditional (NoLatent + NoSmooth)
                    'Trad_Mean': np.mean(boot_stdgf_nosmooth[val]),
                    'Trad_Low': pct(boot_stdgf_nosmooth[val], 2.5),
                    'Trad_High': pct(boot_stdgf_nosmooth[val], 97.5),
                    # Cox PH
                    'Cox_Mean': np.mean(boot_cox[val]),
                    'Cox_Low': pct(boot_cox[val], 2.5),
                    'Cox_High': pct(boot_cox[val], 97.5),
                })

    df_strat = pd.DataFrame(res_strat)
    df_strat.to_csv(os.path.join(OUTPUT_DIR, 'Table_Stratified.csv'), index=False)
    df_comp = pd.DataFrame(comp_results)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, 'Table_Comparison_Severe.csv'), index=False)

    # -----------------------------------------------------------------
    # 3. Figures
    # -----------------------------------------------------------------
    print("\n🎨 Generating Figures...")

    if not df_comp.empty:
        # =============================================================
        # Figure 2A: Clinical Paradigm Shift
        # Cox (linear, no confounding correction)
        # vs Traditional G-formula (noisy, no latent, no smooth)
        # vs Proposed (full model)
        # =============================================================
        fig, ax = plt.subplots(figsize=(10, 7))

        plot_smooth_ci(ax, df_comp['MP'], df_comp['Cox_Mean'],
                       df_comp['Cox_Low'], df_comp['Cox_High'],
                       '#2ca02c', 'Cox PH (Clinical Standard)',
                       linestyle='-.')
        plot_smooth_ci(ax, df_comp['MP'], df_comp['Trad_Mean'],
                       df_comp['Trad_Low'], df_comp['Trad_High'],
                       '#7f7f7f', 'Traditional G-formula',
                       linestyle='--')
        plot_smooth_ci(ax, df_comp['MP'], df_comp['HMM_Mean'],
                       df_comp['HMM_Low'], df_comp['HMM_High'],
                       '#d62728', f'Proposed Model (λ={LAMBDA_SMOOTH})')

        ax.set_xlabel('Mechanical Power (J/min)')
        ax.set_ylabel('30-Day Mortality Risk (%)')
        ax.set_title('Figure 2A. Causal Dose-Response of Mechanical Power in Severe ARDS')
        ax.legend(loc='upper left')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'Figure2A_Main_Result.png'))
        plt.close(fig)

        # =============================================================
        # Figure 2B: Ablation Study
        #
        #                 No Smooth          Smooth
        # No Latent   Traditional (gray)    StdGF+S (blue)
        # Latent      HMM no-S (orange)     Proposed (red)
        # =============================================================
        fig, ax = plt.subplots(figsize=(10, 7))

        # (1) Traditional: No Latent, No Smooth — gray thin dotted
        ax.plot(df_comp['MP'], df_comp['Trad_Mean'],
                color='#7f7f7f', linestyle=':', linewidth=1.2,
                label='No Latent, No Smooth (Traditional)')

        # (2) StdGF + Smooth: No Latent, Smooth — blue dashed
        ax.plot(df_comp['MP'], df_comp['SGF_S_Mean'],
                color='#1f77b4', linestyle='--', linewidth=1.8,
                label='No Latent, Smooth')

        # (3) HMM no-smooth: Latent, No Smooth — orange dotted
        ax.plot(df_comp['MP'], df_comp['HMM_NS_Mean'],
                color='#ff7f0e', linestyle=':', linewidth=1.8,
                label='Latent, No Smooth')

        # (4) Proposed: Latent, Smooth — red solid thick
        ax.plot(df_comp['MP'], df_comp['HMM_Mean'],
                color='#d62728', linestyle='-', linewidth=2.5,
                label='Latent + Smooth (Proposed)')

        ax.set_xlabel('Mechanical Power (J/min)')
        ax.set_ylabel('30-Day Mortality Risk (%)')
        ax.set_title('Figure 2B. Ablation: Contribution of Latent State and Smoothness')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'Figure2B_Ablation.png'))
        plt.close(fig)

    # Figure 3: Subgroup
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'Severe': '#d62728', 'Moderate': '#ff7f0e', 'Mild': '#2ca02c'}
    for grp in indices.keys():
        sub = df_strat[df_strat['Group'] == grp]
        if sub.empty:
            continue
        plot_smooth_ci(ax, sub['MP'], sub['Mean'], sub['Low'], sub['High'],
                       colors[grp], grp)
    ax.axvline(17, color='gray', ls='--', label='17 J/min Reference')
    ax.legend()
    ax.set_xlabel('Mechanical Power (J/min)')
    ax.set_ylabel('30-Day Mortality Risk (%)')
    ax.set_title('Figure 3. Dose-Response Stratified by ARDS Severity')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'Figure3_Subgroup.png'))
    plt.close(fig)

    # Table 2
    if not df_comp.empty:
        ref_idx = (df_comp['MP'] - 17).abs().idxmin()
        ref_risk = df_comp.loc[ref_idx, 'HMM_Mean']
        opt_idx = df_comp['HMM_Mean'].idxmin()
        opt_mp = df_comp.loc[opt_idx, 'MP']
        opt_risk = df_comp.loc[opt_idx, 'HMM_Mean']
        ard = ref_risk - opt_risk
        pd.DataFrame([{
            'Comparison': f'{opt_mp:.1f}J vs 17J',
            'Risk_Opt': f"{opt_risk:.1f}%",
            'Risk_Ref': f"{ref_risk:.1f}%",
            'RR': f"{opt_risk / (ref_risk + 1e-10):.2f}",
            'ARD': f"{ard:.1f}%",
            'NNT': f"{100 / (ard + 1e-10):.1f}" if ard > 0 else "N/A",
        }]).to_csv(os.path.join(OUTPUT_DIR, 'Table2_Clinical_Impact.csv'), index=False)

    # -----------------------------------------------------------------
    # 4. Diagnostics
    # -----------------------------------------------------------------
    if cox_full is not None:
        summary = cox_full.get_summary()
        if summary is not None:
            summary.to_csv(os.path.join(OUTPUT_DIR, 'Table_CoxPH_Summary.csv'))
            print("\n📋 Cox PH Coefficients:")
            print(summary[['coef', 'exp(coef)', 'p']].to_string())

    if model_full is not None:
        print("\n📋 Proposed Model Parameters:")
        for k, v in model_full.get_parameters().items():
            if isinstance(v, list):
                print(f"   {k}: [{', '.join(f'{x:.3f}' for x in v[:5])}...]")
            else:
                print(f"   {k}: {v:.4f}")

    # -----------------------------------------------------------------
    # 5. Supplementary: λ Sensitivity Analysis
    # -----------------------------------------------------------------
    # Fit proposed model at multiple λ values on Severe subgroup
    # to show that the U-shape finding is robust to regularization choice.
    #
    # Output:
    #   - Supp_Table_Lambda_Sensitivity.csv (optimal MP + risk at each λ)
    #   - Supp_Figure_Lambda_Sensitivity.png (overlay plot)
    # -----------------------------------------------------------------
    if len(idx_sev) >= 50:
        print("\n📋 Supplementary: λ Sensitivity Analysis...")
        lambdas = [0.005, 0.01, 0.02, 0.05, 0.1]
        lambda_colors = {0.005: '#9467bd', 0.01: '#1f77b4', 0.02: '#d62728',
                         0.05: '#ff7f0e', 0.1: '#2ca02c'}

        S_sev = S[idx_sev]
        L_sev = L_dyn[idx_sev]
        C_sev = C_static[idx_sev]
        Y_sev = Y[idx_sev]
        M_sev = mask[idx_sev]

        lambda_results = []  # for table
        fig_supp, ax_supp = plt.subplots(figsize=(10, 7))

        for lam in lambdas:
            print(f"   λ={lam}...")
            m_lam = ContinuousHMM(n_dyn, n_static, N_BINS)
            m_lam.fit(S_sev, L_sev, C_sev, Y_sev, M_sev,
                      n_epochs=N_EPOCHS, lambda_smooth=lam, lr=LR)

            risks = []
            for val in targets:
                t_s = standardize_mp(val, sc_S)
                r = m_lam.simulate_gformula(L_sev[:, 0, :], C_sev, t_s)[-1] * 100
                risks.append(r)

            risks = np.array(risks)

            # Plot
            lw = 2.5 if lam == LAMBDA_SMOOTH else 1.2
            ls = '-' if lam == LAMBDA_SMOOTH else '--'
            label = f'λ={lam}' + (' (primary)' if lam == LAMBDA_SMOOTH else '')
            ax_supp.plot(targets, risks, color=lambda_colors[lam],
                         linestyle=ls, linewidth=lw, label=label)

            # Table row
            opt_idx_lam = np.argmin(risks)
            lambda_results.append({
                'Lambda': lam,
                'Optimal_MP': targets[opt_idx_lam],
                'Min_Risk': risks[opt_idx_lam],
                'Risk_at_17J': risks[np.argmin(np.abs(targets - 17))],
                'Risk_at_10J': risks[np.argmin(np.abs(targets - 10))],
                'Risk_at_25J': risks[np.argmin(np.abs(targets - 25))],
            })

        ax_supp.set_xlabel('Mechanical Power (J/min)')
        ax_supp.set_ylabel('30-Day Mortality Risk (%)')
        ax_supp.set_title('Supplementary: Sensitivity to Smoothness Penalty (λ)')
        ax_supp.legend(loc='upper left')
        fig_supp.tight_layout()
        fig_supp.savefig(os.path.join(OUTPUT_DIR, 'Supp_Figure_Lambda_Sensitivity.png'))
        plt.close(fig_supp)

        df_lambda = pd.DataFrame(lambda_results)
        df_lambda.to_csv(os.path.join(OUTPUT_DIR, 'Supp_Table_Lambda_Sensitivity.csv'),
                         index=False)
        print(df_lambda.to_string(index=False))

    print(f"\n✅ Complete. Results → {OUTPUT_DIR}")


if __name__ == "__main__":
    run_all()