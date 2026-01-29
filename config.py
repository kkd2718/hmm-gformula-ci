"""
config.py - Simulation Study Configuration (v3.2)

한국인 코호트(KoGES) 및 최신 통계 반영 버전 (정밀 보정)
- 2023 KNHANES 흡연율: 남성 36.9%, 여성 8.3%
- 2022 KDCA 심뇌혈관질환: 10년 누적 5-10%

v3.2 Changes:
- 흡연 지속성(alpha_S) 하향 조정 (2.5 → 2.0)
- CVD 기초 위험(beta_0) 하향 조정 (-5.5 → -6.5)
- [NEW] 시간 효과(Aging Effect) 파라미터 추가
  - alpha_time: 시간 경과에 따른 흡연 확률 감소
  - beta_time: 시간 경과에 따른 CVD 위험 증가
"""

import numpy as np

# =============================================================================
# 재현성 설정
# =============================================================================
SEED = 42
np.random.seed(SEED)

# =============================================================================
# 공변량 설정 (Covariates)
# =============================================================================
COVARIATE_NAMES = ['age', 'sex', 'bmi']
N_COVARIATES = 3

IDX_AGE = 0
IDX_SEX = 1
IDX_BMI = 2

# =============================================================================
# 데이터 생성 파라미터 (TRUE VALUES - Ground Truth)
# 한국인 역학 통계 기반 정밀 보정 (v3.2)
# =============================================================================

# -----------------------------------------------------------------------------
# Hidden State Transition Model:
# Z_t = ψ*Z_{t-1} + γ_S*S_t + γ_C*C_t + γ_G*G + γ_L*L + γ_{GS}*(G×S_t) + γ_{GC}*(G×C_t) + ε
#
# [v3.2] Z 증가 속도 완화: gamma_S, gamma_C 하향 조정
# -----------------------------------------------------------------------------
TRUE_PARAMS_HIDDEN_STATE = {
    'psi': 0.6,           # Z 자기회귀 (0.7 → 0.6: 누적 효과 완화)
    'gamma_S': 0.15,      # 흡연 → Z (0.25 → 0.15)
    'gamma_C': 0.08,      # Pack-years → Z (0.15 → 0.08)
    'gamma_G': 0.15,      # PRS → Z
    'gamma_GS': 0.10,     # G×S interaction
    'gamma_GC': 0.05,     # G×C interaction
    'sigma_Z': 0.1,       # Noise
    
    # 공변량 효과 [Age, Sex, BMI]
    'gamma_L': [0.10, 0.08, 0.08],
}

# -----------------------------------------------------------------------------
# Outcome Model:
# logit(Y_t) = β_0 + β_Z*Z_t + β_S*S_t + β_G*G + β_L*L + β_{GS}*(G×S_t) + β_time*t
#
# [v3.2] 보정:
# - beta_0: -5.5 → -6.5 (기초 위험도 대폭 하향)
# - beta_Z, beta_S: 하향 조정
# - [NEW] beta_time: +0.05 (노화에 따른 위험 증가)
#
# Target: 10년 누적 CVD 5-10%, 남/여 비 ~2.9
# -----------------------------------------------------------------------------
TRUE_PARAMS_OUTCOME = {
    'beta_0': -6.5,       # Baseline (연간 ~0.15% 기초 위험도)
    'beta_Z': 0.4,        # Z → Y (0.6 → 0.4)
    'beta_S': 0.15,       # S → Y 직접 효과 (0.25 → 0.15)
    'beta_G': 0.10,       # G → Y
    'beta_GS': 0.25,      # G×S interaction (0.35 → 0.25)
    
    # 공변량 효과 [Age, Sex, BMI]
    # Sex: log(2.9) ≈ 1.06
    'beta_L': [0.20, 1.06, 0.15],
    
    # [NEW] 시간 효과 (Aging)
    'beta_time': 0.05,    # 매년 CVD 위험 자연 증가 (노화)
}

# -----------------------------------------------------------------------------
# Exposure (Smoking) Model:
# logit(S_t) = α_0 + α_S*S_{t-1} + α_Z*Z_{t-1} + α_G*G + α_L*L + α_time*t
#
# [v3.2] 보정:
# - alpha_S: 2.5 → 2.0 (지속성 완화 → 금연 용이)
# - [NEW] alpha_time: -0.05 (나이 들수록 금연 증가)
#
# Target: 남성 35-40%, 여성 8-10% (시간 평균)
# -----------------------------------------------------------------------------
TRUE_PARAMS_EXPOSURE = {
    'alpha_0': -2.40,     # Baseline (여성 기준 ~8.3%)
    'alpha_S': 2.0,       # 흡연 지속성 (2.5 → 2.0: 금연 용이)
    'alpha_Z': 0.10,      # Z → S confounding (0.15 → 0.10)
    'alpha_G': 0.03,      # G → S (0.05 → 0.03)
    
    # 공변량 효과 [Age, Sex, BMI]
    # Sex: 남성 효과 (36.9% vs 8.3%)
    'alpha_L': [-0.02, 1.86, 0.03],
    
    # [NEW] 시간 효과 (Aging)
    'alpha_time': -0.05,  # 매년 흡연 확률 감소 (나이 들수록 금연)
}

# Pack-years 계산
PACK_YEAR_INCREMENT = 1.0

# =============================================================================
# 기본 데이터 설정
# =============================================================================
DEFAULT_DATA_PARAMS = {
    'n_samples': 20000,
    'n_time': 10,
}

# =============================================================================
# 모델 학습 설정
# =============================================================================
TRAINING_PARAMS = {
    'n_epochs': 200,
    'n_epochs_large': 150,
    'learning_rate': 0.01,
    'em_max_iter': 100,
    'em_tol': 1e-6,
    'batch_size': None,
    'large_n_threshold': 50000,
}

# =============================================================================
# 실험 설정
# =============================================================================

EXP1_EFFECT_SIZES = {
    'Null': {'gamma_GS': 0.0, 'gamma_GC': 0.0, 'beta_GS': 0.0},
    'Weak': {'gamma_GS': 0.03, 'gamma_GC': 0.02, 'beta_GS': 0.10},
    'Moderate': {'gamma_GS': 0.10, 'gamma_GC': 0.05, 'beta_GS': 0.25},
    'Strong': {'gamma_GS': 0.20, 'gamma_GC': 0.10, 'beta_GS': 0.40},
}

EXP2_SAMPLE_SIZES = [5000, 10000, 20000, 50000, 100000]

EXP3_MISSPEC_SCENARIOS = {
    'Correct': {'fit_interaction': True, 'fit_pack_years': True, 'use_hmm': True},
    'No_Interaction': {'fit_interaction': False, 'fit_pack_years': True, 'use_hmm': True},
    'No_PackYears': {'fit_interaction': True, 'fit_pack_years': False, 'use_hmm': True},
    'No_HMM': {'fit_interaction': True, 'fit_pack_years': True, 'use_hmm': False},
}

# =============================================================================
# Monte Carlo 설정
# =============================================================================
MC_PARAMS = {
    'n_simulations': 200,
    'n_simulations_quick': 30,
    'n_bootstrap': 200,
    'confidence_level': 0.95,
}

GFORMULA_PARAMS = {
    'n_monte_carlo': 1000,
    'interventions': {
        'natural': 'No intervention',
        'always_smoke': 'Always smoke',
        'never_smoke': 'Never smoke',
        'quit_at_t3': 'Quit at year 3',
        'quit_at_t5': 'Quit at year 5',
    },
}

# =============================================================================
# 출력 설정
# =============================================================================
OUTPUT_DIR = './results'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# =============================================================================
# Helper functions
# =============================================================================

def get_modified_params(base_params: dict, modifications: dict) -> dict:
    modified = base_params.copy()
    modified.update(modifications)
    return modified


def get_n_epochs(n_samples: int) -> int:
    if n_samples >= TRAINING_PARAMS.get('large_n_threshold', 50000):
        return TRAINING_PARAMS.get('n_epochs_large', 150)
    return TRAINING_PARAMS.get('n_epochs', 200)


def validate_config():
    """설정값 검증 및 예상 통계 출력"""
    import math
    
    print("=" * 70)
    print("Configuration Validation (v3.2 - Calibrated)")
    print("=" * 70)
    
    # 초기 흡연율 (t=0, 이전 흡연 없음)
    female_init = 1 / (1 + math.exp(-TRUE_PARAMS_EXPOSURE['alpha_0']))
    male_init = 1 / (1 + math.exp(-(TRUE_PARAMS_EXPOSURE['alpha_0'] + TRUE_PARAMS_EXPOSURE['alpha_L'][IDX_SEX])))
    
    # 흡연 지속율 (t=5, 이전 흡연 있음)
    t = 5
    female_persist = 1 / (1 + math.exp(-(TRUE_PARAMS_EXPOSURE['alpha_0'] + 
                                          TRUE_PARAMS_EXPOSURE['alpha_S'] * 1 +
                                          TRUE_PARAMS_EXPOSURE['alpha_time'] * t)))
    male_persist = 1 / (1 + math.exp(-(TRUE_PARAMS_EXPOSURE['alpha_0'] + 
                                        TRUE_PARAMS_EXPOSURE['alpha_S'] * 1 +
                                        TRUE_PARAMS_EXPOSURE['alpha_L'][IDX_SEX] +
                                        TRUE_PARAMS_EXPOSURE['alpha_time'] * t)))
    
    print(f"\n[Smoking - Initial (t=0)] Target: M 37%, F 8%")
    print(f"  Female: {female_init*100:.1f}%")
    print(f"  Male: {male_init*100:.1f}%")
    
    print(f"\n[Smoking - Persistence (t=5, prev smoker)]")
    print(f"  Female: {female_persist*100:.1f}%")
    print(f"  Male: {male_persist*100:.1f}%")
    print(f"  (alpha_time={TRUE_PARAMS_EXPOSURE['alpha_time']} → 금연 촉진)")
    
    # CVD 위험
    baseline_cvd = 1 / (1 + math.exp(-TRUE_PARAMS_OUTCOME['beta_0']))
    cvd_t5 = 1 / (1 + math.exp(-(TRUE_PARAMS_OUTCOME['beta_0'] + TRUE_PARAMS_OUTCOME['beta_time'] * 5)))
    cvd_t10 = 1 / (1 + math.exp(-(TRUE_PARAMS_OUTCOME['beta_0'] + TRUE_PARAMS_OUTCOME['beta_time'] * 10)))
    
    male_cvd_t0 = 1 / (1 + math.exp(-(TRUE_PARAMS_OUTCOME['beta_0'] + TRUE_PARAMS_OUTCOME['beta_L'][IDX_SEX])))
    male_cvd_t10 = 1 / (1 + math.exp(-(TRUE_PARAMS_OUTCOME['beta_0'] + 
                                        TRUE_PARAMS_OUTCOME['beta_L'][IDX_SEX] +
                                        TRUE_PARAMS_OUTCOME['beta_time'] * 10)))
    
    print(f"\n[CVD Risk - Baseline (no risk factors)] Target: 10y cumulative 5-10%")
    print(f"  t=0: {baseline_cvd*100:.3f}%")
    print(f"  t=5: {cvd_t5*100:.3f}%")
    print(f"  t=10: {cvd_t10*100:.3f}%")
    print(f"  (beta_time={TRUE_PARAMS_OUTCOME['beta_time']} → 노화 효과)")
    
    print(f"\n[CVD Risk - Male]")
    print(f"  t=0: {male_cvd_t0*100:.3f}%")
    print(f"  t=10: {male_cvd_t10*100:.3f}%")
    print(f"  Sex Ratio (M/F at t=0): {male_cvd_t0/baseline_cvd:.2f}")
    
    # 10년 누적 추정 (단순 합산)
    cumulative_female = sum([1/(1+math.exp(-(TRUE_PARAMS_OUTCOME['beta_0'] + TRUE_PARAMS_OUTCOME['beta_time']*t))) 
                             for t in range(10)])
    cumulative_male = sum([1/(1+math.exp(-(TRUE_PARAMS_OUTCOME['beta_0'] + 
                                            TRUE_PARAMS_OUTCOME['beta_L'][IDX_SEX] +
                                            TRUE_PARAMS_OUTCOME['beta_time']*t))) 
                           for t in range(10)])
    
    print(f"\n[Estimated 10y Cumulative CVD (baseline, no Z/S effect)]")
    print(f"  Female: ~{cumulative_female*100:.1f}%")
    print(f"  Male: ~{cumulative_male*100:.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    validate_config()