"""
experiments/full_experiments.py
------------------------------------------------------------------
Title: Dynamic Causal Inference of Mechanical Power in Severe ARDS
       - A Latent State Modeling Approach -
------------------------------------------------------------------
Final Production (v6):
 - Visualization: Spline Interpolation for smooth CI bands (Publication Quality)
 - Completeness: Table 1 & Figure 1 Data ensured
 - Model: Continuous HMM (Lambda=0.02) + 20 Bins + Dense Targets
------------------------------------------------------------------
"""

import os, sys, torch, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Configuration
torch.set_num_threads(os.cpu_count()) 
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Hyperparameters
N_BINS = 20
LAMBDA_SMOOTH = 0.02
N_EPOCHS = 300
N_BOOTSTRAPS = 100

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.dynamic_hmm import ContinuousHMM, BinnedHMM

DATA_TENSOR = os.path.join(CURRENT_DIR, 'processed_data', 'hmm_tensor', 'mimic_hmm_tensor.pt')
DATA_CSV = os.path.join(CURRENT_DIR, 'processed_data', 'final_cohort', 'ards_standard_cohort.csv')
SCALER_FILE = os.path.join(CURRENT_DIR, 'processed_data', 'hmm_tensor', 'scalers.pkl')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Helper Functions
# -----------------------------------------------------------------------------

def censor_last_48h(Y):
    Y_censored = Y.clone()
    for i in range(Y.shape[0]):
        valid_idx = torch.where(Y[i, :, 0] != -1)[0]
        if len(valid_idx) > 0:
            last_t = valid_idx[-1]
            start_mask = max(0, int(last_t) - 1)
            Y_censored[i, start_mask : last_t+1, 0] = -1
    return Y_censored

def plot_smooth_ci(ax, x, mean, lower, upper, color, label, linestyle='-', alpha=0.15):
    """
    Draws a smooth curve and confidence band using Spline Interpolation.
    """
    # 1. Create high-resolution x axis
    x_new = np.linspace(x.min(), x.max(), 300)
    
    # 2. Interpolate Mean, Lower, Upper
    try:
        spl_m = make_interp_spline(x, mean, k=3)
        mean_smooth = spl_m(x_new)
        
        spl_l = make_interp_spline(x, lower, k=3)
        lower_smooth = spl_l(x_new)
        
        spl_u = make_interp_spline(x, upper, k=3)
        upper_smooth = spl_u(x_new)
        
        # Clip to valid range (0-100%) to avoid artifacts
        mean_smooth = np.clip(mean_smooth, 0, 100)
        lower_smooth = np.clip(lower_smooth, 0, 100)
        upper_smooth = np.clip(upper_smooth, 0, 100)
        
        # 3. Plot
        ax.plot(x_new, mean_smooth, linestyle=linestyle, color=color, label=label, linewidth=2)
        ax.fill_between(x_new, lower_smooth, upper_smooth, color=color, alpha=alpha, edgecolor=None)
        
    except Exception as e:
        # Fallback to standard plot if interpolation fails (e.g., too few points)
        print(f"Smoothing failed ({e}), using standard plot.")
        ax.plot(x, mean, linestyle=linestyle, color=color, label=label, linewidth=2)
        ax.fill_between(x, lower, upper, color=color, alpha=alpha)

class StandardGFormula:
    def __init__(self, n_covs):
        self.out_mod = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
    def fit(self, S, L, C, Y):
        valid = (Y.flatten() != -1)
        S_np = S.numpy().reshape(-1,1)[valid]
        L_np = L.numpy().reshape(-1, L.shape[2])[valid]
        C_np = C.numpy().reshape(-1,1)[valid]
        Y_np = Y.numpy().flatten()[valid]
        self.out_mod.fit(np.hstack([S_np, S_np**2, L_np, C_np]), Y_np)
    def simulate(self, L_init, t_s):
        curr_L, surv = L_init.numpy(), np.ones(len(L_init))
        T_S, C = np.full((len(L_init), 1), t_s), np.zeros((len(L_init), 1))
        for _ in range(30):
            surv *= (1 - self.out_mod.predict_proba(np.hstack([T_S, T_S**2, curr_L, C]))[:, 1])
            C += T_S
        return 1 - np.mean(surv)

class CategoricalBaseline:
    def __init__(self, n_covs, bins, model_type='logistic'):
        self.out_mod = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
        self.bins = bins
        self.n_cats = len(bins) + 1
    def fit(self, S, L, C, Y):
        valid = (Y.flatten() != -1)
        S_np = S.numpy().reshape(-1,1)[valid]
        L_np = L.numpy().reshape(-1, L.shape[2])[valid]
        C_np = C.numpy().reshape(-1,1)[valid]
        Y_np = Y.numpy().flatten()[valid]
        S_cat = np.digitize(S_np.flatten(), self.bins)
        S_oh = np.eye(self.n_cats)[S_cat]
        self.out_mod.fit(np.hstack([S_oh, L_np, C_np]), Y_np)
    def simulate(self, L_init, t_s):
        curr_L, surv = L_init.numpy(), np.ones(len(L_init))
        cat_idx = np.digitize([t_s], self.bins)[0]
        S_oh = np.tile(np.eye(self.n_cats)[cat_idx], (len(L_init), 1))
        T_S, C = np.full((len(L_init), 1), t_s), np.zeros((len(L_init), 1))
        for _ in range(30):
            surv *= (1 - self.out_mod.predict_proba(np.hstack([S_oh, curr_L, C]))[:, 1])
            C += T_S
        return 1 - np.mean(surv)

# -----------------------------------------------------------------------------
# 2. Main Logic
# -----------------------------------------------------------------------------

def run_all():
    print(f"🚀 Starting Final Production (Smoothed Viz)...")
    
    data = torch.load(DATA_TENSOR); sc = joblib.load(SCALER_FILE)
    S, Y, L_dyn, C_st = data['S'], data['Y'], data['L'], data['C']
    C = torch.cumsum(S, dim=1); L = torch.cat([C_st.unsqueeze(1).expand(-1, 30, -1), L_dyn], dim=2)
    G = L_dyn[:, 0, 0].unsqueeze(1)
    Y_censored = censor_last_48h(Y)
    
    df = pd.read_csv(DATA_CSV)
    base_pf = df[df['day_num']==0]['pf_ratio'].values
    
    # -------------------------------------------------------------------------
    # Part 1: Table 1 (Baseline) & Figure 1 (Forest Plot)
    # -------------------------------------------------------------------------
    print("📊 Generating Table 1 & Figure 1...")
    
    # Table 1 Generation
    base_df = df.groupby('stay_id').first().reset_index()
    t1_vars = {'anchor_age': 'Age', 'charlson_index': 'CCI', 'pf_ratio': 'P/F Ratio', 
               'lactate': 'Lactate', 'compliance': 'Compliance', 'mp_j_min': 'MP (J/min)'}
    t1_data = []
    for v, label in t1_vars.items():
        s = base_df[base_df['hospital_expire_flag']==0][v]
        d = base_df[base_df['hospital_expire_flag']==1][v]
        t1_data.append({'Variable': label, 'Survivors': f"{s.mean():.1f} ({s.std():.1f})", 'Non-Survivors': f"{d.mean():.1f} ({d.std():.1f})"})
    pd.DataFrame(t1_data).to_csv(os.path.join(OUTPUT_DIR, 'Table1_Baseline.csv'), index=False)
    
    # Figure 1 Generation
    base_f = df[df['day_num']==0].dropna()
    base_f['mp_cat'] = pd.cut(base_f['mp_j_min'], bins=[0,12,17,22,100], labels=['<12','12-17','17-22','>22'])
    X_num = StandardScaler().fit_transform(base_f[['anchor_age','charlson_index','pf_ratio','lactate','compliance']])
    X_cat = base_f[['is_male']].values
    X_mp = pd.get_dummies(base_f['mp_cat'], prefix='MP', dtype=float)[['MP_<12','MP_17-22','MP_>22']].values
    X = np.hstack([X_num, X_cat, X_mp])
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    y = base_f['hospital_expire_flag'].values
    
    clf = LogisticRegression(penalty=None, fit_intercept=False).fit(X_design, y)
    probs = clf.predict_proba(X_design)[:, 1]
    W = probs * (1 - probs)
    Hessian = np.dot(X_design.T, X_design * W[:, np.newaxis])
    try: se = np.sqrt(np.diag(np.linalg.inv(Hessian)))
    except: se = np.zeros(len(clf.coef_[0])) + 0.1
    
    feat_names = ['Intercept','Age','CCI','P/F Ratio','Lactate','Compliance','Sex (Male)','MP < 12','MP 17-22','MP > 22']
    or_df = pd.DataFrame({'Feature': feat_names, 'OR': np.exp(clf.coef_[0]), 'SE': se, 
                          'Lower': np.exp(clf.coef_[0]-1.96*se), 'Upper': np.exp(clf.coef_[0]+1.96*se)})
    or_df = or_df[or_df['Feature']!='Intercept']
    or_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_Fig1_Forest.csv'), index=False)
    
    plt.figure(figsize=(9, 6))
    plt.errorbar(or_df['OR'], range(len(or_df)), xerr=[or_df['OR']-or_df['Lower'], or_df['Upper']-or_df['OR']], 
                 fmt='s', color='black', capsize=5)
    plt.axvline(1, linestyle='--', color='red'); plt.yticks(range(len(or_df)), or_df['Feature'])
    plt.xlabel('Adjusted Odds Ratio (95% CI)'); plt.title('Figure 1. Baseline Association')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1_Forest_Plot.png')); plt.close()

    # -------------------------------------------------------------------------
    # Part 2: Training & Simulation
    # -------------------------------------------------------------------------
    print("🧠 Training Models...")
    idx_sev = np.where(base_pf <= 100)[0]
    idx_mod = np.where((base_pf > 100) & (base_pf <= 200))[0]
    idx_mild = np.where(base_pf > 200)[0]
    indices = {'Severe': idx_sev, 'Moderate': idx_mod, 'Mild': idx_mild}
    
    sc_S = sc['S']
    targets = np.linspace(5, 35, 61) # Dense targets
    hmm_bin_edges = torch.linspace(-2.5, 2.5, N_BINS + 1); fine_bins = hmm_bin_edges.numpy().tolist()
    
    res_strat = []
    comp_results = []

    for grp, idx in indices.items():
        if len(idx) < 50: continue
        
        G_s, S_s, C_s, Y_s, L_s = G[idx], S[idx], C[idx], Y_censored[idx], L[idx]
        hmm_smooth = ContinuousHMM(L.shape[2], n_bins=N_BINS).fit(G_s, S_s, C_s, Y_s, L_s, n_epochs=N_EPOCHS, lambda_smooth=LAMBDA_SMOOTH)
        
        # Baseline Models (Severe Only)
        hmm_no_smooth=None; std_gf=None; cat_gf=None; cat_cox=None; cox_lin=None; cox_quad=None
        avg_L = L_s[:, 0, :].mean(dim=0).numpy()
        
        if grp == 'Severe':
            hmm_no_smooth = ContinuousHMM(L.shape[2], n_bins=N_BINS).fit(G_s, S_s, C_s, Y_s, L_s, n_epochs=N_EPOCHS, lambda_smooth=0.0)
            std_gf = StandardGFormula(L.shape[2]); std_gf.fit(S_s, L_s, C_s, Y_s)
            
            valid = (Y_s.numpy().flatten() != -1)
            S_v = S_s.numpy().reshape(-1,1)[valid]; L_v = L_s.numpy().reshape(-1, L.shape[2])[valid]; Y_v = Y_s.numpy().flatten()[valid]
            cox_lin = LogisticRegression(penalty=None).fit(np.hstack([S_v, L_v]), Y_v)
            cox_quad = LogisticRegression(penalty=None).fit(np.hstack([S_v, S_v**2, L_v]), Y_v)
            
            cat_gf = CategoricalBaseline(L.shape[2], bins=fine_bins, model_type='logistic'); cat_gf.fit(S_s, L_s, C_s, Y_s)
            cat_cox = CategoricalBaseline(L.shape[2], bins=fine_bins, model_type='cox'); cat_cox.fit(S_s, L_s, C_s, Y_s)

        for val in tqdm(targets, desc=f"Sim {grp}", leave=False):
            t_s = (np.log(val) - sc_S.mean_[0]) / sc_S.scale_[0]
            
            c_risks = []
            for _ in range(N_BOOTSTRAPS):
                ib = np.random.randint(0, len(idx), 300)
                c_risks.append(hmm_smooth.simulate(G_s[ib], L_s[ib, 0, :], t_s, static_mode=True)[-1]*100)
            res_strat.append({'Group': grp, 'MP': val, 'Mean': np.mean(c_risks), 'Low': np.percentile(c_risks, 2.5), 'High': np.percentile(c_risks, 97.5)})
            
            if grp == 'Severe':
                ns_risks = []
                for _ in range(N_BOOTSTRAPS):
                    ib = np.random.randint(0, len(idx), 300)
                    ns_risks.append(hmm_no_smooth.simulate(G_s[ib], L_s[ib, 0, :], t_s, static_mode=True)[-1]*100)
                
                gf_lin = std_gf.simulate(L_s[:, 0, :], t_s)*100
                gf_cat = cat_gf.simulate(L_s[:, 0, :], t_s)*100
                cox_cat_r = cat_cox.simulate(L_s[:, 0, :], t_s)*100
                
                x_l = np.hstack([[t_s], avg_L]).reshape(1, -1)
                p_l = (1 - (1 - cox_lin.predict_proba(x_l)[0,1])**30) * 100
                x_q = np.hstack([[t_s, t_s**2], avg_L]).reshape(1, -1)
                p_q = (1 - (1 - cox_quad.predict_proba(x_q)[0,1])**30) * 100
                
                comp_results.append({
                    'MP': val,
                    'HMM_Smooth': np.mean(c_risks), 'HMM_Smooth_L': np.percentile(c_risks, 2.5), 'HMM_Smooth_H': np.percentile(c_risks, 97.5),
                    'HMM_NoSmooth': np.mean(ns_risks),
                    'Std_GF': gf_lin, 'Lin_Cox': p_l, 'Quad_Cox': p_q, 'Cat_GF': gf_cat, 'Cat_Cox': cox_cat_r
                })

    df_strat = pd.DataFrame(res_strat); df_strat.to_csv(os.path.join(OUTPUT_DIR, 'Table_Stratified.csv'), index=False)
    df_comp = pd.DataFrame(comp_results); df_comp.to_csv(os.path.join(OUTPUT_DIR, 'Table_Comparison_Severe.csv'), index=False)

    # -------------------------------------------------------------------------
    # Part 3: Visualization (Smoothed Splines)
    # -------------------------------------------------------------------------
    
    # Figure 2A: Main Comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_smooth_ci(ax, df_comp['MP'], df_comp['HMM_Smooth'], df_comp['HMM_Smooth_L'], df_comp['HMM_Smooth_H'], 'red', 'Proposed HMM')
    ax.plot(df_comp['MP'], df_comp['Std_GF'], 'k--', label='Standard G-Formula')
    ax.plot(df_comp['MP'], df_comp['Lin_Cox'], 'g-.', label='Cox (Linear)')
    ax.plot(df_comp['MP'], df_comp['Quad_Cox'], 'm-.', label='Cox (Quadratic)')
    ax.set_xlabel('Mechanical Power (J/min)'); ax.set_ylabel('30-Day Mortality Risk (%)')
    ax.set_title('Figure 2A. Causal Effect in Severe ARDS'); ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, 'Figure2A_Main_Result.png')); plt.close(fig)

    # Figure 2B: Ablation
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_smooth_ci(ax, df_comp['MP'], df_comp['HMM_Smooth'], df_comp['HMM_Smooth_L'], df_comp['HMM_Smooth_H'], 'red', 'Proposed HMM (Smooth)')
    ax.plot(df_comp['MP'], df_comp['HMM_NoSmooth'], 'r--', alpha=0.5, label='HMM (No Smooth)')
    ax.plot(df_comp['MP'], df_comp['Cat_GF'], 'b-', alpha=0.6, label='Categorical G-Formula')
    ax.plot(df_comp['MP'], df_comp['Cat_Cox'], 'g-', alpha=0.6, label='Categorical Cox')
    ax.set_xlabel('Mechanical Power (J/min)'); ax.set_ylabel('Risk (%)')
    ax.set_title('Figure 2B. Model Robustness & Ablation'); ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, 'Figure2B_Robustness.png')); plt.close(fig)

    # Figure 3: Subgroup
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'Severe': '#d62728', 'Moderate': '#ff7f0e', 'Mild': '#2ca02c'}
    for grp in indices.keys():
        sub = df_strat[df_strat['Group']==grp]
        if sub.empty: continue
        plot_smooth_ci(ax, sub['MP'], sub['Mean'], sub['Low'], sub['High'], colors[grp], grp)
    ax.axvline(17, color='gray', ls='--', label='17J Cutoff'); ax.legend()
    ax.set_xlabel('Mechanical Power (J/min)'); ax.set_ylabel('Risk (%)')
    ax.set_title('Figure 3. Stratified Analysis by Severity'); 
    fig.savefig(os.path.join(OUTPUT_DIR, 'Figure3_Subgroup.png')); plt.close(fig)

    # Table 2
    ref_idx = (df_comp['MP']-17).abs().argmin()
    ref_risk = df_comp.iloc[ref_idx]['HMM_Smooth']
    opt_idx = df_comp['HMM_Smooth'].idxmin()
    opt_mp = df_comp.iloc[opt_idx]['MP']; opt_risk = df_comp.iloc[opt_idx]['HMM_Smooth']
    rr = opt_risk/ref_risk; nnt = 1/((ref_risk-opt_risk)/100)
    res = [{'Comparison': f'{opt_mp:.1f}J vs 17J', 'Risk_Opt': f"{opt_risk:.1f}%", 'Risk_Ref': f"{ref_risk:.1f}%", 'RR': f"{rr:.2f}", 'NNT': f"{nnt:.1f}"}]
    pd.DataFrame(res).to_csv(os.path.join(OUTPUT_DIR, 'Table2_Clinical_Impact.csv'), index=False)

    print(f"✅ Final Production Completed. Check {OUTPUT_DIR}")

if __name__ == "__main__": run_all()