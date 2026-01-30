"""
config.py - Simulation Study Configuration (v3.3)

KoGES 실제 추적 기간 반영: 20년 시뮬레이션
- 2023 KNHANES 흡연율: 남성 36.9%, 여성 8.3%
- 2022 KDCA 심뇌혈관질환: 20년 누적 5-10%

v3.3 Changes:
- n_time: 10 → 20 (KoGES 추적 기간 반영)
- alpha_S: 1.5 → 1.2 (금연 용이성 증가)
- alpha_time: -0.03 → -0.05 (우하향 트렌드 강화)
- beta_0: -6.5 → -7.2 (20년 누적 발생률 조정)
- beta_time: 0.05 → 0.04 (과대 추정 방지)
"""

import numpy as np

# =============================================================================
# 재현성 설정
# =============================================================================
SEED = 42
np.random.seed(SEED)

# =============================================================================
# 공변량 설정 (Covariates)
# L = [Age (standardized), Sex (0=F, 1=M), BMI (standardized)]
# =============================================================================
COVARIATE_NAMES = ['age', 'sex', 'bmi']
N_COVARIATES = 3

IDX_AGE = 0
IDX_SEX = 1
IDX_BMI = 2

# =============================================================================
# 데이터 생성 파라미터 (TRUE VALUES - Ground Truth)
# 20년 시뮬레이션 보정 (v3.3)
# =============================================================================

# -----------------------------------------------------------------------------
# Hidden State Transition Model:
# Z_t = ψ*Z_{t-1} + γ_S*S_t + γ_C*C_t + γ_G*G + γ_L*L + γ_{GS}*(G×S_t) + γ_{GC}*(G×C_t) + ε
# -----------------------------------------------------------------------------
TRUE_PARAMS_HIDDEN_STATE = {
    'psi': 0.6,           # Z 자기회귀
    'gamma_S': 0.15,      # 흡연 → Z
    'gamma_C': 0.08,      # Pack-years → Z
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
# [v3.3 보정] 20년 시뮬레이션용:
# - beta_0: -6.5 → -7.2 (기초 위험도 하향)
# - beta_time: 0.05 → 0.04 (노화 증가 속도 미세 조정)
#
# Target: 20년 누적 CVD 5-10%
# -----------------------------------------------------------------------------
TRUE_PARAMS_OUTCOME = {
    'beta_0': -7.2,       # Baseline (연간 ~0.07% 기초 위험도)
    'beta_Z': 0.4,        # Z → Y
    'beta_S': 0.15,       # S → Y 직접 효과
    'beta_G': 0.10,       # G → Y
    'beta_GS': 0.25,      # G×S interaction
    
    # 공변량 효과 [Age, Sex, BMI]
    # Sex: log(2.9) ≈ 1.06 → 남성 ~2.9배 위험
    'beta_L': [0.20, 1.06, 0.15],
    
    # 시간 효과 (Aging)
    'beta_time': 0.04,    # 매년 CVD 위험 증가 (v3.3: 0.05 → 0.04)
}

# -----------------------------------------------------------------------------
# Exposure (Smoking) Model:
# logit(S_t) = α_0 + α_S*S_{t-1} + α_Z*Z_{t-1} + α_G*G + α_L*L + α_time*t
#
# [v3.3 보정] 20년 우하향 곡선:
# - alpha_S: 1.5 → 1.2 (지속성 완화)
# - alpha_time: -0.03 → -0.05 (감소 트렌드 강화)
# - alpha_L: [-0.08, 1.9, 0.0] (Age 감소, Sex 남성 증가, BMI 무효과)
#
# Target: 남성 t=0 ~60%, t=19 ~35%
# -----------------------------------------------------------------------------
TRUE_PARAMS_EXPOSURE = {
    'alpha_0': -2.40,     # Baseline (여성 기준)
    'alpha_S': 1.2,       # 흡연 지속성 (v3.3: 1.5 → 1.2)
    'alpha_Z': 0.10,      # Z → S confounding
    'alpha_G': 0.03,      # G → S
    
    # 공변량 효과 [Age, Sex, BMI]
    # Age: -0.08 (나이 들수록 금연)
    # Sex: 1.9 (남성 흡연율 높음)
    # BMI: 0.0 (무효과)
    'alpha_L': [-0.08, 1.9, 0.0],
    
    # 시간 효과 (Aging)
    'alpha_time': -0.05,  # 매년 흡연 확률 감소 (v3.3: -0.03 → -0.05)
}

# Pack-years 계산
PACK_YEAR_INCREMENT = 1.0

# =============================================================================
# 기본 데이터 설정 (v3.3: 20년 시뮬레이션)
# =============================================================================
DEFAULT_DATA_PARAMS = {
    'n_samples': 20000,
    'n_time': 20,         # v3.3: 10 → 20 (KoGES 추적 기간)
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
        'quit_at_t5': 'Quit at year 5',
        'quit_at_t10': 'Quit at year 10',
        'quit_at_t15': 'Quit at year 15',
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
    """설정값 검증 및 예상 통계 출력 (20년 시뮬레이션)"""
    import math
    
    n_time = DEFAULT_DATA_PARAMS['n_time']
    
    print("=" * 70)
    print(f"Configuration Validation (v3.3 - {n_time}-Year Simulation)")
    print("=" * 70)
    
    # =========================================================================
    # 흡연율 시뮬레이션
    # =========================================================================
    print(f"\n[Smoking Rate Trajectory] (Target: M 60%→35%, F 8%→5%)")
    
    # 남성 (이전 흡연 가정)
    print("\n  Male (previous smoker):")
    for t in [0, 5, 10, 15, 19]:
        logit = (TRUE_PARAMS_EXPOSURE['alpha_0'] + 
                 TRUE_PARAMS_EXPOSURE['alpha_S'] * 1 +  # 이전 흡연
                 TRUE_PARAMS_EXPOSURE['alpha_L'][IDX_SEX] +  # 남성
                 TRUE_PARAMS_EXPOSURE['alpha_time'] * t)
        prob = 1 / (1 + math.exp(-logit))
        print(f"    t={t:2d}: {prob*100:.1f}%")
    
    # 여성 (이전 흡연 가정)
    print("\n  Female (previous smoker):")
    for t in [0, 5, 10, 15, 19]:
        logit = (TRUE_PARAMS_EXPOSURE['alpha_0'] + 
                 TRUE_PARAMS_EXPOSURE['alpha_S'] * 1 +
                 TRUE_PARAMS_EXPOSURE['alpha_time'] * t)
        prob = 1 / (1 + math.exp(-logit))
        print(f"    t={t:2d}: {prob*100:.1f}%")
    
    # =========================================================================
    # CVD 발생률 시뮬레이션
    # =========================================================================
    print(f"\n[CVD Risk Trajectory] (Target: 20y cumulative 5-10%)")
    
    # Baseline (no risk factors)
    print("\n  Baseline (female, non-smoker, Z=0):")
    cumulative = 0
    for t in range(n_time):
        logit = TRUE_PARAMS_OUTCOME['beta_0'] + TRUE_PARAMS_OUTCOME['beta_time'] * t
        prob = 1 / (1 + math.exp(-logit))
        cumulative += prob
        if t in [0, 5, 10, 15, 19]:
            print(f"    t={t:2d}: annual {prob*100:.3f}%, cumulative ~{cumulative*100:.2f}%")
    
    # Male
    print("\n  Male (non-smoker, Z=0):")
    cumulative = 0
    for t in range(n_time):
        logit = (TRUE_PARAMS_OUTCOME['beta_0'] + 
                 TRUE_PARAMS_OUTCOME['beta_L'][IDX_SEX] +
                 TRUE_PARAMS_OUTCOME['beta_time'] * t)
        prob = 1 / (1 + math.exp(-logit))
        cumulative += prob
        if t in [0, 5, 10, 15, 19]:
            print(f"    t={t:2d}: annual {prob*100:.3f}%, cumulative ~{cumulative*100:.2f}%")
    
    # Male smoker (with Z effect)
    print("\n  Male smoker (with moderate Z~1):")
    cumulative = 0
    for t in range(n_time):
        logit = (TRUE_PARAMS_OUTCOME['beta_0'] + 
                 TRUE_PARAMS_OUTCOME['beta_L'][IDX_SEX] +
                 TRUE_PARAMS_OUTCOME['beta_Z'] * 1.0 +  # Z = 1
                 TRUE_PARAMS_OUTCOME['beta_S'] * 1 +    # current smoker
                 TRUE_PARAMS_OUTCOME['beta_time'] * t)
        prob = 1 / (1 + math.exp(-logit))
        cumulative += prob
        if t in [0, 5, 10, 15, 19]:
            print(f"    t={t:2d}: annual {prob*100:.3f}%, cumulative ~{cumulative*100:.2f}%")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "-" * 70)
    print("Key Parameters (v3.3):")
    print("-" * 70)
    print(f"  n_time: {n_time} years")
    print(f"  alpha_S (smoking persistence): {TRUE_PARAMS_EXPOSURE['alpha_S']}")
    print(f"  alpha_time (yearly smoking change): {TRUE_PARAMS_EXPOSURE['alpha_time']}")
    print(f"  beta_0 (baseline CVD logit): {TRUE_PARAMS_OUTCOME['beta_0']}")
    print(f"  beta_time (yearly CVD increase): {TRUE_PARAMS_OUTCOME['beta_time']}")
    print("=" * 70)


if __name__ == "__main__":
    validate_config()