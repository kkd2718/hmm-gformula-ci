"""
real_data_adapter.py - 실제 코호트 데이터 적용을 위한 어댑터

KoGES, KCPS-II, UKBB 등 실제 데이터를 모델에 적용하기 위한 인터페이스

주요 기능:
1. 다양한 데이터 형식 지원 (pandas DataFrame, numpy array)
2. 결측치 처리
3. 변수 변환 및 표준화
4. PRS 계산 (외부 도구 연동)
5. Pack-years 계산
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class CohortData:
    """실제 코호트 데이터 컨테이너"""
    G: torch.Tensor          # PRS: (N, 1)
    S: torch.Tensor          # Smoking status: (N, T, 1)
    C: torch.Tensor          # Pack-years: (N, T, 1)
    Y: torch.Tensor          # Outcome: (N, T, 1)
    L: torch.Tensor          # Covariates: (N, K)
    
    # 메타데이터
    subject_ids: Optional[np.ndarray] = None
    time_points: Optional[np.ndarray] = None
    covariate_names: Optional[List[str]] = None
    
    # 결측치 마스크
    missing_mask: Optional[torch.Tensor] = None
    
    @property
    def n_samples(self) -> int:
        return self.G.shape[0]
    
    @property
    def n_time(self) -> int:
        return self.S.shape[1]
    
    @property
    def n_covariates(self) -> int:
        return self.L.shape[1] if self.L is not None else 0


class RealDataAdapter:
    """
    실제 코호트 데이터를 모델 입력 형식으로 변환
    
    지원 데이터셋:
    - KoGES (Korean Genome and Epidemiology Study)
    - KCPS-II (Korean Cancer Prevention Study-II)
    - UKBB (UK Biobank)
    - TWB (Taiwan Biobank)
    - BBJ (BioBank Japan)
    """
    
    def __init__(
        self,
        outcome_col: str = 'cvd_event',
        smoking_col: str = 'smoking_status',
        cigarettes_per_day_col: str = 'cigarettes_per_day',
        prs_col: str = 'prs_cvd',
        time_col: str = 'wave',
        id_col: str = 'subject_id',
        covariate_cols: List[str] = None,
    ):
        """
        Args:
            outcome_col: CVD 발생 여부 컬럼명
            smoking_col: 흡연 상태 컬럼명 (0/1 or categorical)
            cigarettes_per_day_col: 하루 흡연량 컬럼명
            prs_col: PRS 컬럼명
            time_col: 시점 컬럼명
            id_col: 피험자 ID 컬럼명
            covariate_cols: 공변량 컬럼명 리스트 (예: ['age', 'sex', 'bmi'])
        """
        self.outcome_col = outcome_col
        self.smoking_col = smoking_col
        self.cigarettes_per_day_col = cigarettes_per_day_col
        self.prs_col = prs_col
        self.time_col = time_col
        self.id_col = id_col
        self.covariate_cols = covariate_cols or ['age', 'sex', 'bmi']
        
        # 표준화 파라미터 저장
        self._standardization_params = {}
    
    def _standardize(self, x: np.ndarray, name: str, fit: bool = True) -> np.ndarray:
        """변수 표준화 (z-score)"""
        if fit:
            mean = np.nanmean(x)
            std = np.nanstd(x)
            if std < 1e-8:
                std = 1.0
            self._standardization_params[name] = {'mean': mean, 'std': std}
        else:
            params = self._standardization_params.get(name, {'mean': 0, 'std': 1})
            mean, std = params['mean'], params['std']
        
        return (x - mean) / std
    
    def _compute_pack_years(
        self,
        smoking_status: np.ndarray,
        cigarettes_per_day: np.ndarray,
        years_per_wave: float = 1.0,
    ) -> np.ndarray:
        """
        Pack-years 계산
        
        Pack-year = (하루 흡연량 / 20) × 흡연 년수
        """
        # 결측치 처리
        cpd = np.nan_to_num(cigarettes_per_day, nan=0.0)
        smoking = np.nan_to_num(smoking_status, nan=0.0)
        
        # 각 wave의 pack-year 기여분
        packs_per_day = cpd / 20.0  # 1갑 = 20개비
        wave_contribution = packs_per_day * smoking * years_per_wave
        
        # 누적 pack-years
        pack_years = np.cumsum(wave_contribution, axis=1)
        
        return pack_years
    
    def from_long_format(
        self,
        df: pd.DataFrame,
        fit_standardization: bool = True,
    ) -> CohortData:
        """
        Long format 데이터프레임을 CohortData로 변환
        
        Long format: 각 행이 (subject, time) 조합
        
        Args:
            df: pandas DataFrame in long format
            fit_standardization: True면 표준화 파라미터 계산
        """
        # 피험자와 시점 추출
        subjects = df[self.id_col].unique()
        times = sorted(df[self.time_col].unique())
        
        n_subjects = len(subjects)
        n_times = len(times)
        
        # ID 매핑
        subject_to_idx = {s: i for i, s in enumerate(subjects)}
        time_to_idx = {t: i for i, t in enumerate(times)}
        
        # 초기화
        G = np.zeros((n_subjects, 1))
        S = np.zeros((n_subjects, n_times, 1))
        Y = np.zeros((n_subjects, n_times, 1))
        cpd = np.zeros((n_subjects, n_times))  # cigarettes per day
        L = np.zeros((n_subjects, len(self.covariate_cols)))
        missing_mask = np.ones((n_subjects, n_times, 1))  # 1 = observed, 0 = missing
        
        # 데이터 채우기
        for _, row in df.iterrows():
            subj_idx = subject_to_idx[row[self.id_col]]
            time_idx = time_to_idx[row[self.time_col]]
            
            # PRS (시간 불변)
            if self.prs_col in row and pd.notna(row[self.prs_col]):
                G[subj_idx, 0] = row[self.prs_col]
            
            # Smoking
            if self.smoking_col in row:
                val = row[self.smoking_col]
                if pd.isna(val):
                    missing_mask[subj_idx, time_idx, 0] = 0
                else:
                    S[subj_idx, time_idx, 0] = float(val > 0)  # 0/1로 변환
            
            # Cigarettes per day
            if self.cigarettes_per_day_col in row:
                val = row[self.cigarettes_per_day_col]
                cpd[subj_idx, time_idx] = 0 if pd.isna(val) else val
            
            # Outcome
            if self.outcome_col in row:
                val = row[self.outcome_col]
                if pd.isna(val):
                    missing_mask[subj_idx, time_idx, 0] = 0
                else:
                    Y[subj_idx, time_idx, 0] = float(val > 0)
            
            # Covariates (baseline만 사용 - 첫 시점)
            if time_idx == 0:
                for cov_idx, cov_col in enumerate(self.covariate_cols):
                    if cov_col in row and pd.notna(row[cov_col]):
                        L[subj_idx, cov_idx] = row[cov_col]
        
        # Pack-years 계산
        C = self._compute_pack_years(S[:, :, 0], cpd)
        C = C[:, :, np.newaxis]
        
        # 표준화
        G = self._standardize(G, 'G', fit=fit_standardization)
        for i, cov_col in enumerate(self.covariate_cols):
            L[:, i] = self._standardize(L[:, i], cov_col, fit=fit_standardization)
        
        # Tensor 변환
        return CohortData(
            G=torch.tensor(G, dtype=torch.float32),
            S=torch.tensor(S, dtype=torch.float32),
            C=torch.tensor(C, dtype=torch.float32),
            Y=torch.tensor(Y, dtype=torch.float32),
            L=torch.tensor(L, dtype=torch.float32),
            subject_ids=subjects,
            time_points=np.array(times),
            covariate_names=self.covariate_cols,
            missing_mask=torch.tensor(missing_mask, dtype=torch.float32),
        )
    
    def from_wide_format(
        self,
        df: pd.DataFrame,
        time_varying_cols: Dict[str, List[str]],
        fit_standardization: bool = True,
    ) -> CohortData:
        """
        Wide format 데이터프레임을 CohortData로 변환
        
        Wide format: 각 행이 한 피험자, 컬럼이 시점별 변수
        
        Args:
            df: pandas DataFrame in wide format
            time_varying_cols: 시점별 컬럼명 매핑
                예: {
                    'smoking': ['smoke_w1', 'smoke_w2', ..., 'smoke_w10'],
                    'outcome': ['cvd_w1', 'cvd_w2', ..., 'cvd_w10'],
                }
        """
        n_subjects = len(df)
        n_times = len(time_varying_cols.get('smoking', []))
        
        # PRS
        G = df[self.prs_col].values.reshape(-1, 1) if self.prs_col in df.columns else np.zeros((n_subjects, 1))
        
        # Time-varying variables
        smoking_cols = time_varying_cols.get('smoking', [])
        outcome_cols = time_varying_cols.get('outcome', [])
        cpd_cols = time_varying_cols.get('cigarettes_per_day', smoking_cols)
        
        S = np.zeros((n_subjects, n_times, 1))
        Y = np.zeros((n_subjects, n_times, 1))
        cpd = np.zeros((n_subjects, n_times))
        
        for t, (s_col, y_col, c_col) in enumerate(zip(smoking_cols, outcome_cols, cpd_cols)):
            if s_col in df.columns:
                S[:, t, 0] = df[s_col].fillna(0).values
            if y_col in df.columns:
                Y[:, t, 0] = df[y_col].fillna(0).values
            if c_col in df.columns:
                cpd[:, t] = df[c_col].fillna(0).values
        
        # Pack-years
        C = self._compute_pack_years(S[:, :, 0], cpd)
        C = C[:, :, np.newaxis]
        
        # Covariates
        L = np.zeros((n_subjects, len(self.covariate_cols)))
        for i, cov_col in enumerate(self.covariate_cols):
            if cov_col in df.columns:
                L[:, i] = df[cov_col].fillna(df[cov_col].mean()).values
        
        # 표준화
        G = self._standardize(G, 'G', fit=fit_standardization)
        for i, cov_col in enumerate(self.covariate_cols):
            L[:, i] = self._standardize(L[:, i], cov_col, fit=fit_standardization)
        
        return CohortData(
            G=torch.tensor(G, dtype=torch.float32),
            S=torch.tensor(S, dtype=torch.float32),
            C=torch.tensor(C, dtype=torch.float32),
            Y=torch.tensor(Y, dtype=torch.float32),
            L=torch.tensor(L, dtype=torch.float32),
            subject_ids=df[self.id_col].values if self.id_col in df.columns else None,
            covariate_names=self.covariate_cols,
        )
    
    def handle_missing(
        self,
        data: CohortData,
        method: str = 'complete_case',
    ) -> CohortData:
        """
        결측치 처리
        
        Args:
            data: CohortData
            method: 처리 방법
                - 'complete_case': 완전 케이스만 사용
                - 'locf': Last Observation Carried Forward
                - 'mean': 평균 대체
        """
        if method == 'complete_case':
            # 결측치가 없는 피험자만 선택
            if data.missing_mask is not None:
                complete_mask = data.missing_mask.all(dim=1).squeeze()
                idx = complete_mask.nonzero().squeeze()
                
                return CohortData(
                    G=data.G[idx],
                    S=data.S[idx],
                    C=data.C[idx],
                    Y=data.Y[idx],
                    L=data.L[idx],
                    subject_ids=data.subject_ids[idx.numpy()] if data.subject_ids is not None else None,
                    time_points=data.time_points,
                    covariate_names=data.covariate_names,
                )
        
        elif method == 'locf':
            # Last Observation Carried Forward
            S_filled = data.S.clone()
            for t in range(1, data.n_time):
                mask = data.missing_mask[:, t, :] == 0
                S_filled[:, t, :] = torch.where(mask, S_filled[:, t-1, :], S_filled[:, t, :])
            
            return CohortData(
                G=data.G,
                S=S_filled,
                C=data.C,  # C는 S에서 다시 계산해야 함
                Y=data.Y,
                L=data.L,
                subject_ids=data.subject_ids,
                time_points=data.time_points,
                covariate_names=data.covariate_names,
            )
        
        return data


# =============================================================================
# 예시: KoGES 데이터 적용
# =============================================================================

def example_koges_analysis():
    """
    KoGES 데이터 분석 예시 (실제 데이터 필요)
    """
    print("=" * 60)
    print("Example: KoGES Data Analysis Pipeline")
    print("=" * 60)
    
    # 1. 데이터 로드 (가상)
    print("\n1. Loading KoGES data...")
    print("   [실제 분석 시 여기에 데이터 로드 코드]")
    # df = pd.read_csv('/path/to/koges_data.csv')
    
    # 2. 어댑터 설정
    print("\n2. Setting up data adapter...")
    adapter = RealDataAdapter(
        outcome_col='mi_stroke',          # 심근경색/뇌졸중
        smoking_col='sm_present',         # 현재 흡연 여부
        cigarettes_per_day_col='sm_cpd',  # 하루 흡연량
        prs_col='prs_cvd_adjusted',       # PRS (인구 구조 보정)
        time_col='wave',
        id_col='pid',
        covariate_cols=['age', 'sex', 'bmi', 'alcohol'],
    )
    
    # 3. 데이터 변환
    print("\n3. Converting to model format...")
    print("   [실제 분석 시: data = adapter.from_long_format(df)]")
    
    # 4. 모델 학습
    print("\n4. Fitting HMM-gFormula model...")
    print("   [실제 분석 시: model.fit(data.G, data.S, data.C, data.Y, L=data.L)]")
    
    # 5. g-formula 시뮬레이션
    print("\n5. Running g-formula simulations...")
    print("   [실제 분석 시: causal = estimate_causal_effect(model, data.G, L=data.L)]")
    
    # 6. 결과 보고
    print("\n6. Results summary...")
    print("   - Causal Risk Difference (smoke vs. no smoke)")
    print("   - Causal Risk Ratio")
    print("   - Interaction effects by PRS quantile")
    
    print("\n" + "=" * 60)
    print("Pipeline complete. Replace placeholders with actual data.")
    print("=" * 60)


if __name__ == "__main__":
    example_koges_analysis()