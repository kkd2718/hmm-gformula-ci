"""
data_generator.py - Data Generating Process (v3.3)

한국인 코호트 특성 반영 + 시간 효과(Aging) 포함

v3.3 Changes:
- [UPDATE] validate_dgp: 20년 시뮬레이션 지원 (동적 시점 표시)
- [UPDATE] stats 키 이름 변경 (t9 → t_last, 10y → n_time 기반)

v3.2 Changes:
- [NEW] alpha_time: 시간 경과에 따른 흡연 확률 감소
- [NEW] beta_time: 시간 경과에 따른 CVD 위험 증가
- 수식에 `+ alpha_time * t`, `+ beta_time * t` 항 추가

Models:
1. Hidden State: Z_t = ψ*Z_{t-1} + γ_S*S_t + γ_C*C_t + γ_G*G + γ_L*L + γ_{GS}*(G×S_t) + γ_{GC}*(G×C_t) + ε
2. Exposure: logit(S_t) = α_0 + α_S*S_{t-1} + α_Z*Z_{t-1} + α_G*G + α_L*L + α_time*t
3. Outcome: logit(Y_t) = β_0 + β_Z*Z_t + β_S*S_t + β_G*G + β_L*L + β_{GS}*(G×S_t) + β_time*t
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from config import (
    TRUE_PARAMS_HIDDEN_STATE,
    TRUE_PARAMS_OUTCOME,
    TRUE_PARAMS_EXPOSURE,
    PACK_YEAR_INCREMENT,
    N_COVARIATES,
    COVARIATE_NAMES,
    IDX_AGE, IDX_SEX, IDX_BMI,
    SEED,
)


@dataclass
class SimulatedData:
    """시뮬레이션 데이터 컨테이너"""
    G: torch.Tensor          # Genetic risk score (PRS): (N, 1)
    S: torch.Tensor          # Smoking status over time: (N, T, 1)
    C: torch.Tensor          # Cumulative pack-years: (N, T, 1)
    Y: torch.Tensor          # CVD outcome: (N, T, 1)
    Z_true: torch.Tensor     # True latent state: (N, T, 1)
    L: torch.Tensor          # Covariates: (N, K) - [Age, Sex, BMI]
    at_risk: torch.Tensor    # At-risk indicator: (N, T, 1)
    
    @property
    def n_samples(self) -> int:
        return self.G.shape[0]
    
    @property
    def n_time(self) -> int:
        return self.S.shape[1]
    
    @property
    def n_covariates(self) -> int:
        return self.L.shape[1]
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        return {
            'G': self.G.numpy(),
            'S': self.S.numpy(),
            'C': self.C.numpy(),
            'Y': self.Y.numpy(),
            'Z_true': self.Z_true.numpy(),
            'L': self.L.numpy(),
            'at_risk': self.at_risk.numpy(),
        }
    
    def get_sex(self) -> torch.Tensor:
        return self.L[:, IDX_SEX:IDX_SEX+1]
    
    def get_male_mask(self) -> torch.Tensor:
        return (self.L[:, IDX_SEX] > 0.5)
    
    def get_female_mask(self) -> torch.Tensor:
        return (self.L[:, IDX_SEX] <= 0.5)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid"""
    return torch.sigmoid(x.clamp(-20, 20))


def generate_covariates(
    n_samples: int,
    sex_ratio: float = 0.5,
    age_mean: float = 55.0,
    age_std: float = 10.0,
    bmi_mean: float = 24.0,
    bmi_std: float = 3.5,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """공변량 생성: Age, Sex, BMI"""
    if seed is not None:
        torch.manual_seed(seed)
    
    age_raw = torch.randn(n_samples, 1) * age_std + age_mean
    age_raw = age_raw.clamp(40, 75)
    age_std_tensor = (age_raw - age_mean) / age_std
    
    sex = torch.bernoulli(torch.full((n_samples, 1), sex_ratio))
    
    bmi_raw = torch.randn(n_samples, 1) * bmi_std + bmi_mean
    bmi_raw = bmi_raw.clamp(16, 40)
    bmi_std_tensor = (bmi_raw - bmi_mean) / bmi_std
    
    L = torch.cat([age_std_tensor, sex, bmi_std_tensor], dim=1)
    return L


def generate_prs(n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
    """PRS (Polygenic Risk Score) 생성"""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(n_samples, 1)


def compute_linear_covariate_effect(L: torch.Tensor, coefficients: List[float]) -> torch.Tensor:
    """공변량의 선형 효과 계산"""
    coef_tensor = torch.tensor(coefficients, dtype=torch.float32).view(1, -1)
    effect = (L * coef_tensor).sum(dim=1, keepdim=True)
    return effect


def generate_synthetic_data(
    n_samples: int,
    n_time: int,
    params_hidden: Optional[Dict] = None,
    params_outcome: Optional[Dict] = None,
    params_exposure: Optional[Dict] = None,
    sex_ratio: float = 0.5,
    survival_outcome: bool = True,
    seed: Optional[int] = None,
) -> SimulatedData:
    """
    한국인 특성 반영 + 시간 효과(Aging) 포함 합성 데이터 생성
    
    [v3.2] 시간 효과 추가:
    - Smoking: logit(S_t) += alpha_time * t
    - CVD: logit(Y_t) += beta_time * t
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 파라미터 설정
    p_h = params_hidden or TRUE_PARAMS_HIDDEN_STATE.copy()
    p_o = params_outcome or TRUE_PARAMS_OUTCOME.copy()
    p_e = params_exposure or TRUE_PARAMS_EXPOSURE.copy()
    
    # 시간 효과 파라미터 (없으면 0)
    alpha_time = p_e.get('alpha_time', 0.0)
    beta_time = p_o.get('beta_time', 0.0)
    
    # Baseline variables
    G = generate_prs(n_samples, seed)
    L = generate_covariates(n_samples, sex_ratio=sex_ratio, seed=seed)
    
    # Storage
    S_list, C_list, Y_list, Z_list, at_risk_list = [], [], [], [], []
    
    # Initialize
    Z_prev = torch.zeros(n_samples, 1)
    
    # 초기 흡연 상태 (t=0): 성별 + 시간 효과
    sex_effect = L[:, IDX_SEX:IDX_SEX+1] * p_e['alpha_L'][IDX_SEX]
    initial_smoke_logit = p_e['alpha_0'] + sex_effect + alpha_time * 0
    initial_smoke_prob = sigmoid(initial_smoke_logit)
    S_prev = torch.bernoulli(initial_smoke_prob)
    
    C_prev = torch.zeros(n_samples, 1)
    event_occurred = torch.zeros(n_samples, 1, dtype=torch.bool)
    
    with torch.no_grad():
        for t in range(n_time):
            # At-risk status
            at_risk_t = (~event_occurred).float()
            
            # =================================================================
            # Step 1: Exposure (Smoking) S_t
            # logit(S_t) = α_0 + α_S*S_{t-1} + α_Z*Z_{t-1} + α_G*G + α_L*L + α_time*t
            # =================================================================
            L_effect_S = compute_linear_covariate_effect(L, p_e['alpha_L'])
            
            logit_S = (
                p_e['alpha_0'] +
                p_e['alpha_S'] * S_prev +
                p_e['alpha_Z'] * Z_prev +
                p_e['alpha_G'] * G +
                L_effect_S +
                alpha_time * t  # [NEW] 시간 효과
            )
            prob_S = sigmoid(logit_S)
            S_curr = torch.bernoulli(prob_S)
            
            # =================================================================
            # Step 2: Pack-years C_t
            # =================================================================
            C_curr = C_prev + S_curr * PACK_YEAR_INCREMENT
            
            # =================================================================
            # Step 3: Hidden State Z_t
            # =================================================================
            L_effect_Z = compute_linear_covariate_effect(L, p_h['gamma_L'])
            noise = torch.randn(n_samples, 1) * p_h['sigma_Z']
            
            Z_curr = (
                p_h['psi'] * Z_prev +
                p_h['gamma_S'] * S_curr +
                p_h['gamma_C'] * C_curr +
                p_h['gamma_G'] * G +
                L_effect_Z +
                p_h['gamma_GS'] * (G * S_curr) +
                p_h['gamma_GC'] * (G * C_curr) +
                noise
            )
            
            # =================================================================
            # Step 4: Outcome Y_t
            # logit(Y_t) = β_0 + β_Z*Z_t + β_S*S_t + β_G*G + β_L*L + β_{GS}*(G×S_t) + β_time*t
            # =================================================================
            L_effect_Y = compute_linear_covariate_effect(L, p_o['beta_L'])
            
            logit_Y = (
                p_o['beta_0'] +
                p_o['beta_Z'] * Z_curr +
                p_o['beta_S'] * S_curr +
                p_o['beta_G'] * G +
                L_effect_Y +
                p_o['beta_GS'] * (G * S_curr) +
                beta_time * t  # [NEW] 시간 효과 (노화)
            )
            prob_Y = sigmoid(logit_Y)
            Y_curr = torch.bernoulli(prob_Y)
            
            # Survival outcome
            if survival_outcome:
                Y_curr = Y_curr * at_risk_t
                event_occurred = event_occurred | (Y_curr > 0.5)
            
            # Store
            S_list.append(S_curr)
            C_list.append(C_curr)
            Y_list.append(Y_curr)
            Z_list.append(Z_curr)
            at_risk_list.append(at_risk_t)
            
            # Update
            Z_prev = Z_curr
            S_prev = S_curr
            C_prev = C_curr
    
    # Stack: (N, T, 1)
    S = torch.stack(S_list, dim=1)
    C = torch.stack(C_list, dim=1)
    Y = torch.stack(Y_list, dim=1)
    Z_true = torch.stack(Z_list, dim=1)
    at_risk = torch.stack(at_risk_list, dim=1)
    
    return SimulatedData(G=G, S=S, C=C, Y=Y, Z_true=Z_true, L=L, at_risk=at_risk)


def generate_intervention_data(
    G: torch.Tensor,
    L: torch.Tensor,
    intervention: str,
    n_time: int,
    params_hidden: Optional[Dict] = None,
    params_outcome: Optional[Dict] = None,
    params_exposure: Optional[Dict] = None,
    quit_time: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    g-formula 시뮬레이션을 위한 개입 데이터 생성 (시간 효과 포함)
    """
    n_samples = G.shape[0]
    
    p_h = params_hidden or TRUE_PARAMS_HIDDEN_STATE.copy()
    p_o = params_outcome or TRUE_PARAMS_OUTCOME.copy()
    p_e = params_exposure or TRUE_PARAMS_EXPOSURE.copy()
    
    alpha_time = p_e.get('alpha_time', 0.0)
    beta_time = p_o.get('beta_time', 0.0)
    
    S_list, Y_list, Z_list = [], [], []
    
    Z_prev = torch.zeros(n_samples, 1)
    S_prev = torch.zeros(n_samples, 1)
    C_prev = torch.zeros(n_samples, 1)
    
    with torch.no_grad():
        for t in range(n_time):
            # Intervention
            if intervention == 'always_smoke':
                S_curr = torch.ones(n_samples, 1)
            elif intervention == 'never_smoke':
                S_curr = torch.zeros(n_samples, 1)
            elif intervention.startswith('quit_at_t'):
                if t >= quit_time:
                    S_curr = torch.zeros(n_samples, 1)
                else:
                    L_effect_S = compute_linear_covariate_effect(L, p_e['alpha_L'])
                    logit_S = (p_e['alpha_0'] + p_e['alpha_S'] * S_prev + 
                               p_e['alpha_Z'] * Z_prev + p_e['alpha_G'] * G + 
                               L_effect_S + alpha_time * t)
                    S_curr = torch.bernoulli(sigmoid(logit_S))
            else:  # natural
                L_effect_S = compute_linear_covariate_effect(L, p_e['alpha_L'])
                logit_S = (p_e['alpha_0'] + p_e['alpha_S'] * S_prev + 
                           p_e['alpha_Z'] * Z_prev + p_e['alpha_G'] * G + 
                           L_effect_S + alpha_time * t)
                S_curr = torch.bernoulli(sigmoid(logit_S))
            
            C_curr = C_prev + S_curr * PACK_YEAR_INCREMENT
            
            # Hidden state
            L_effect_Z = compute_linear_covariate_effect(L, p_h['gamma_L'])
            Z_curr = (
                p_h['psi'] * Z_prev +
                p_h['gamma_S'] * S_curr +
                p_h['gamma_C'] * C_curr +
                p_h['gamma_G'] * G +
                L_effect_Z +
                p_h['gamma_GS'] * (G * S_curr) +
                p_h['gamma_GC'] * (G * C_curr) +
                torch.randn(n_samples, 1) * p_h['sigma_Z']
            )
            
            # Outcome
            L_effect_Y = compute_linear_covariate_effect(L, p_o['beta_L'])
            logit_Y = (
                p_o['beta_0'] +
                p_o['beta_Z'] * Z_curr +
                p_o['beta_S'] * S_curr +
                p_o['beta_G'] * G +
                L_effect_Y +
                p_o['beta_GS'] * (G * S_curr) +
                beta_time * t  # [NEW] 시간 효과
            )
            prob_Y = sigmoid(logit_Y)
            Y_curr = torch.bernoulli(prob_Y)
            
            S_list.append(S_curr)
            Y_list.append(Y_curr)
            Z_list.append(Z_curr)
            
            Z_prev = Z_curr
            S_prev = S_curr
            C_prev = C_curr
    
    return (
        torch.stack(S_list, dim=1),
        torch.stack(Y_list, dim=1),
        torch.stack(Z_list, dim=1),
    )


def validate_dgp(data: SimulatedData, verbose: bool = True) -> Dict:
    """
    생성된 데이터의 통계량 검증 (한국인 역학 통계와 비교)
    v3.3: 20년 시뮬레이션 지원
    """
    n_time = data.n_time
    male_mask = data.get_male_mask()
    female_mask = data.get_female_mask()
    
    # 성별 흡연율 (시간 평균)
    male_smoke_rate = data.S[male_mask].mean().item()
    female_smoke_rate = data.S[female_mask].mean().item()
    
    # 시점별 흡연율
    male_smoke_by_time = [data.S[male_mask, t, :].mean().item() for t in range(n_time)]
    female_smoke_by_time = [data.S[female_mask, t, :].mean().item() for t in range(n_time)]
    
    # 성별 CVD 발생률
    male_cvd_rate = data.Y[male_mask].mean().item()
    female_cvd_rate = data.Y[female_mask].mean().item()
    
    # 누적 발생률 (전체 기간)
    male_cumulative = 1 - (1 - data.Y[male_mask]).prod(dim=1).mean().item()
    female_cumulative = 1 - (1 - data.Y[female_mask]).prod(dim=1).mean().item()
    
    # 전체 이벤트 수
    total_events = data.Y.sum().item()
    
    # 중간 시점 (10년 또는 n_time//2)
    mid_time = min(9, n_time - 1)
    last_time = n_time - 1
    
    stats = {
        'n_samples': data.n_samples,
        'n_time': n_time,
        'n_male': male_mask.sum().item(),
        'n_female': female_mask.sum().item(),
        'male_smoke_rate': male_smoke_rate,
        'female_smoke_rate': female_smoke_rate,
        'male_smoke_t0': male_smoke_by_time[0],
        'male_smoke_t_last': male_smoke_by_time[last_time],
        'female_smoke_t0': female_smoke_by_time[0],
        'female_smoke_t_last': female_smoke_by_time[last_time],
        'male_cvd_annual': male_cvd_rate,
        'female_cvd_annual': female_cvd_rate,
        'male_cvd_cumulative': male_cumulative,
        'female_cvd_cumulative': female_cumulative,
        'cvd_sex_ratio': male_cvd_rate / (female_cvd_rate + 1e-10),
        'total_cvd_events': total_events,
    }
    
    if verbose:
        print("=" * 70)
        print(f"Data Generation Validation (v3.3 - {n_time}-Year Simulation)")
        print("=" * 70)
        print(f"\n[Sample]")
        print(f"  Total: {stats['n_samples']:,}, Male: {stats['n_male']:,}, Female: {stats['n_female']:,}")
        
        print(f"\n[Smoking Rate] (Target: M ~37%, F ~8%)")
        print(f"  Male: {stats['male_smoke_rate']*100:.1f}% (avg), t0={stats['male_smoke_t0']*100:.1f}%, t{last_time}={stats['male_smoke_t_last']*100:.1f}%")
        print(f"  Female: {stats['female_smoke_rate']*100:.1f}% (avg), t0={stats['female_smoke_t0']*100:.1f}%, t{last_time}={stats['female_smoke_t_last']*100:.1f}%")
        print(f"  → alpha_time 효과: 시간 경과에 따른 흡연율 감소 확인")
        
        print(f"\n[CVD Rate] (Target: {n_time}y cumulative 5-10%)")
        print(f"  Male - Annual: {stats['male_cvd_annual']*100:.2f}%, {n_time}y Cumulative: {stats['male_cvd_cumulative']*100:.1f}%")
        print(f"  Female - Annual: {stats['female_cvd_annual']*100:.2f}%, {n_time}y Cumulative: {stats['female_cvd_cumulative']*100:.1f}%")
        print(f"  Sex Ratio (M/F): {stats['cvd_sex_ratio']:.2f}")
        print(f"  Total CVD Events: {int(stats['total_cvd_events']):,}")
        
        print("=" * 70)
    
    return stats


if __name__ == "__main__":
    print("Testing Korean-calibrated Data Generation (v3.3 - 20yr)...\n")
    
    data = generate_synthetic_data(
        n_samples=20000,
        n_time=20,  # 20년 시뮬레이션
        sex_ratio=0.5,
        survival_outcome=True,
        seed=42,
    )
    
    stats = validate_dgp(data, verbose=True)
    
    print("\nData shapes:")
    print(f"  G: {data.G.shape}")
    print(f"  S: {data.S.shape}")
    print(f"  Y: {data.Y.shape}")
    print(f"  L: {data.L.shape}")