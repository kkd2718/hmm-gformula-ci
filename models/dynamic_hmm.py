"""
models/dynamic_hmm.py — Latent State-Space Model for ARDS (v4.0)
================================================================
Study 1 (hmm_gformula.py)과 구조적으로 일관된 Study 2 모델.

Study 1과의 일관성:
  1. Stochastic latent state (sigma_Z noise)
  2. Forward filter with posterior update from Y
  3. Proper g-formula simulation (covariate evolution + stochastic Z)
  4. Exposure model (continuous MP) for natural course simulation

Study 2 고유 기여:
  - Binned outcome model: 비선형 dose-response (U-shape) 포착
  - Smoothness regularization: 인접 bin 간 strength borrowing
  - CovariateTransitionModel: Ridge 기반 time-varying covariate evolution

Model Equations:
  Latent:   Z_t = ψ·Z_{t-1} + γ_A·A_t + γ_dyn·L_t + γ_static·C + ε_t,  ε~N(0, σ_Z²)
  Outcome:  logit P(Y_t=1) = β_0 + β_Z·Z_t + f(A_t) + β_dyn·L_t + β_static·C + β_time·t
  Covariate: L_t = Ridge(L_{t-1}, A_{t-1})
  
  where f(A_t) = Σ_k β_k · I(A_t ∈ bin_k)  with penalty λ·Σ(β_{k+1}-β_k)²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.linear_model import Ridge
import numpy as np
import copy


# =============================================================================
# Covariate Transition Model (Ridge Regression)
# =============================================================================
class CovariateTransitionModel:
    """
    L_t = f(L_{t-1}, A_{t-1}) via Ridge regression.
    G-formula simulation 시 covariate evolution에 사용.
    """
    def __init__(self, n_covariates):
        self.models = [Ridge(alpha=1.0) for _ in range(n_covariates)]
        self.is_fitted = False

    def fit(self, L_dyn, S):
        """
        Args:
            L_dyn: (N, T, n_cov) dynamic covariates
            S: (N, T, 1) exposure
        """
        L_np = L_dyn.cpu().numpy()
        S_np = S.cpu().numpy()
        N, T, F = L_np.shape

        # X = [L_{t-1}, A_{t-1}], Y = L_t
        X = np.hstack([
            L_np[:, :-1, :].reshape(-1, F),
            S_np[:, :-1, :].reshape(-1, 1)
        ])
        Y = L_np[:, 1:, :].reshape(-1, F)

        # Remove rows with NaN
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X, Y = X[valid], Y[valid]

        for i, model in enumerate(self.models):
            model.fit(X, Y[:, i])
        self.is_fitted = True

    def predict(self, L_curr, S_curr):
        """
        Args:
            L_curr: (N, n_cov) current covariates
            S_curr: (N, 1) current exposure
        Returns:
            (N, n_cov) predicted next covariates
        """
        X = np.hstack([L_curr, S_curr])
        next_L = np.column_stack([m.predict(X) for m in self.models])
        return next_L


# =============================================================================
# ContinuousHMM: Main Model (Consistent with Study 1)
# =============================================================================
class ContinuousHMM(nn.Module):
    """
    Latent State-Space Model with Binned Outcome for ARDS.
    
    Study 1의 HiddenMarkovGFormula와 구조적으로 일관:
    - Stochastic latent state (σ_Z)
    - Forward filter (Kalman-style posterior update from Y)
    - G-formula simulation with covariate evolution
    
    Study 2 추가:
    - Binned piecewise MP effect f(A_t)
    - Smoothness regularization
    """

    def __init__(self, n_dyn_covariates, n_static_covariates, n_bins=15):
        super().__init__()
        self.n_dyn = n_dyn_covariates
        self.n_static = n_static_covariates
        self.n_bins = n_bins

        # Bin edges for standardized log-MP
        self.register_buffer('bin_edges', torch.linspace(-2.5, 2.5, n_bins + 1))

        # --- Latent State Transition ---
        self.psi = nn.Parameter(torch.tensor([0.5]))            # autoregression
        self.gamma_A = nn.Parameter(torch.tensor([0.1]))        # exposure → Z
        self.gamma_dyn = nn.Linear(n_dyn_covariates, 1, bias=False)
        self.gamma_static = nn.Linear(n_static_covariates, 1, bias=False)
        self.log_sigma_Z = nn.Parameter(torch.tensor([-0.5]))   # σ_Z ≈ 0.6 (meaningful Kalman gain)

        nn.init.normal_(self.gamma_dyn.weight, 0, 0.05)
        nn.init.normal_(self.gamma_static.weight, 0, 0.05)

        # --- Outcome Model ---
        self.beta_0 = nn.Parameter(torch.tensor([-4.0]))
        self.beta_Z = nn.Parameter(torch.tensor([0.3]))
        self.beta_bins = nn.Parameter(torch.zeros(n_bins))      # binned MP effect
        self.beta_dyn = nn.Linear(n_dyn_covariates, 1, bias=False)
        self.beta_static = nn.Linear(n_static_covariates, 1, bias=False)
        self.beta_time = nn.Parameter(torch.tensor([0.03]))     # aging/time effect

        nn.init.normal_(self.beta_dyn.weight, 0, 0.1)
        nn.init.normal_(self.beta_static.weight, 0, 0.1)

        # --- Covariate Transition ---
        self.cov_model = CovariateTransitionModel(n_dyn_covariates)

    @property
    def sigma_Z(self):
        return torch.exp(self.log_sigma_Z)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _get_bin_onehot(self, S_cont):
        """Assign continuous exposure to bins."""
        s = S_cont.contiguous().squeeze(-1)
        idx = torch.bucketize(s, self.bin_edges[1:-1])
        return F.one_hot(idx.clamp(0, self.n_bins - 1).long(), self.n_bins).float()

    def _latent_transition(self, Z_prev, A_curr, L_dyn, C_static):
        """
        Z transition: prior mean and variance.
        Returns (Z_mean, Z_var) — Study 1과 동일 구조.
        """
        Z_mean = (self.psi * Z_prev
                  + self.gamma_A * A_curr
                  + self.gamma_dyn(L_dyn)
                  + self.gamma_static(C_static))
        Z_var = self.sigma_Z ** 2
        return Z_mean, Z_var

    def _outcome_probability(self, Z, A_bin, L_dyn, C_static, t=0):
        """P(Y_t=1 | Z_t, A_t, L_t, C)"""
        logit = (self.beta_0
                 + self.beta_Z * Z
                 + (A_bin * self.beta_bins).sum(-1, keepdim=True)
                 + self.beta_dyn(L_dyn)
                 + self.beta_static(C_static)
                 + self.beta_time * (t / 30.0))
        return torch.sigmoid(logit.clamp(-20, 20))

    # -----------------------------------------------------------------
    # Forward Filter (Study 1 일관성: Kalman-style posterior update)
    # -----------------------------------------------------------------
    def forward_filter(self, S, L_dyn, C_static, Y, mask):
        """
        Forward filtering with approximate Kalman update from Y.
        Study 1의 forward_filter와 동일한 구조.
        
        Args:
            S:        (N, T, 1)  exposure
            L_dyn:    (N, T, d)  dynamic covariates
            C_static: (N, p)     static covariates
            Y:        (N, T, 1)  outcome
            mask:     (N, T)     valid indicator
        Returns:
            Z_filtered: (N, T, 1)
            Z_var_filtered: (N, T, 1)
            log_likelihood: scalar
        """
        N, T, _ = S.shape
        device = S.device

        Z_filtered, Z_var_filtered = [], []
        log_likelihoods = []

        Z_mean = torch.zeros(N, 1, device=device)
        Z_var = torch.ones(N, 1, device=device)

        S_bins = self._get_bin_onehot(S)

        for t in range(T):
            A_t = S[:, t, :]
            L_t = L_dyn[:, t, :]
            Y_t = Y[:, t, :]
            m_t = mask[:, t].unsqueeze(1)

            # --- Predict (prior) ---
            Z_pred_mean, Z_pred_var_add = self._latent_transition(
                Z_mean, A_t, L_t, C_static
            )
            Z_pred_var = (self.psi ** 2) * Z_var + Z_pred_var_add

            # --- Update (posterior, using Y) ---
            A_bin_t = S_bins[:, t, :]
            prob_Y = self._outcome_probability(
                Z_pred_mean, A_bin_t, L_t, C_static, t=t
            )

            # Approximate Kalman gain (Study 1과 동일)
            gradient = (Y_t - prob_Y) * self.beta_Z * m_t
            hessian = (self.beta_Z ** 2) * prob_Y * (1 - prob_Y) * m_t + 1e-6

            Z_post_var = 1.0 / (1.0 / (Z_pred_var + 1e-8) + hessian)
            Z_post_mean = Z_pred_mean + Z_post_var * gradient

            # Log-likelihood
            ll = (Y_t * torch.log(prob_Y + 1e-10)
                  + (1 - Y_t) * torch.log(1 - prob_Y + 1e-10)) * m_t
            log_likelihoods.append(ll.sum())

            Z_filtered.append(Z_post_mean)
            Z_var_filtered.append(Z_post_var)

            Z_mean = Z_post_mean
            Z_var = Z_post_var

        return (
            torch.stack(Z_filtered, dim=1),
            torch.stack(Z_var_filtered, dim=1),
            sum(log_likelihoods),
        )

    # -----------------------------------------------------------------
    # Training (EM-style — Study 1 일관성)
    # -----------------------------------------------------------------
    #
    # WHY EM-style?
    # 
    # State-space model의 파라미터 추정에는 두 가지 접근이 있다:
    #
    # (A) End-to-end: Z를 결정론적으로 계산 → loss에서 gradient 흐름
    #     문제: σ_Z가 loss에 기여하지 않음 → 학습 안 됨
    #           시뮬레이션에서 노이즈를 넣으면 학습/추론 불일치
    #
    # (B) EM-style: E-step에서 forward filter로 Z 추정 (σ_Z가 Kalman gain에 영향)
    #              M-step에서 나머지 파라미터 학습
    #     장점: σ_Z가 데이터로부터 간접적으로 학습됨
    #           학습과 시뮬레이션의 stochasticity가 일관
    #
    # σ_Z의 학습 메커니즘:
    #   forward_filter에서 Z_pred_var = ψ² * Z_var + σ_Z²
    #   → σ_Z가 크면: Kalman gain ↑ → Y의 정보를 Z에 더 반영
    #   → σ_Z가 작으면: prior(transition)을 더 신뢰
    #   → 매 E-step마다 σ_Z에 따라 다른 Z가 생성됨
    #   → M-step의 loss가 σ_Z에 간접적으로 의존
    #   → σ_Z는 "Y가 가장 잘 설명되는 noise level"로 수렴
    #
    # Transition params (ψ, γ_A, γ_dyn, γ_static)의 학습:
    #   E-step에서 detach하므로 M-step에서 직접 gradient 없음.
    #   하지만 매 epoch마다 E-step이 갱신된 params로 Z를 재추정하므로
    #   간접적으로 수렴. 이것이 classical EM의 원리.
    #   (Study 1의 hmm_gformula.py와 동일 구조)
    #

    def compute_loss(self, S, L_dyn, C_static, Y, mask, Z_estimated,
                     lambda_smooth=0.02):
        """
        M-step loss: Outcome NLL + Transition MSE + Sigma NLL + Smoothness.
        
        Three gradient paths, each cleanly separated:
        
        (1) Outcome NLL → β_0, β_Z, β_bins, β_dyn, β_static, β_time
        (2) Transition MSE → ψ, γ_A, γ_dyn, γ_static  (no σ_Z in denom)
        (3) Sigma NLL → σ_Z only  (sq_err detached, blocks gradient to ψ/γ)
        
        Why decoupled?
          - MSE for transition params: avoids σ_Z variance collapse
          - NLL with detached sq_err for σ_Z: learns σ_Z = empirical std of
            transition residuals, without destabilizing psi/gamma
        """
        N, T, _ = S.shape
        S_bins = self._get_bin_onehot(S)

        # === (1) Outcome NLL ===
        outcome_nll = torch.tensor(0.0, device=S.device)
        for t in range(T):
            Z_t = Z_estimated[:, t, :]
            A_bin_t = S_bins[:, t, :]
            L_t = L_dyn[:, t, :]
            Y_t = Y[:, t, :]
            m_t = mask[:, t].unsqueeze(1)

            prob_Y = self._outcome_probability(Z_t, A_bin_t, L_t, C_static, t=t)
            bce = (-Y_t * torch.log(prob_Y + 1e-10)
                   - (1 - Y_t) * torch.log(1 - prob_Y + 1e-10))
            outcome_nll = outcome_nll + (bce * m_t).sum() / (m_t.sum() + 1e-10)

        # === (2) Transition MSE + (3) Decoupled Sigma NLL ===
        transition_loss = torch.tensor(0.0, device=S.device)
        sigma_loss = torch.tensor(0.0, device=S.device)
        Z_prev = torch.zeros(N, 1, device=S.device)

        sigma_sq = self.sigma_Z ** 2 + 1e-8

        for t in range(T):
            m_t = mask[:, t].unsqueeze(1)
            Z_target = Z_estimated[:, t, :]

            Z_pred_mean, _ = self._latent_transition(
                Z_prev, S[:, t, :], L_dyn[:, t, :], C_static
            )

            # (2) MSE: gradient → psi, gamma_A, gamma_dyn, gamma_static
            sq_err = (Z_target - Z_pred_mean) ** 2
            transition_loss = transition_loss + (sq_err * m_t).sum() / (m_t.sum() + 1e-10)

            # (3) Sigma NLL: gradient → sigma_Z only
            #     sq_err.detach() blocks gradient from flowing back to psi/gamma
            #     sigma_Z learns to match empirical transition residual variance
            nll_sigma = (0.5 * sq_err.detach() / sigma_sq
                         + 0.5 * torch.log(sigma_sq))
            sigma_loss = sigma_loss + (nll_sigma * m_t).sum() / (m_t.sum() + 1e-10)

            Z_prev = Z_target

        # === Smoothness regularization ===
        diff = self.beta_bins[1:] - self.beta_bins[:-1]
        smooth_loss = lambda_smooth * torch.sum(diff ** 2)

        return outcome_nll + transition_loss + sigma_loss + smooth_loss

    def fit(self, S, L_dyn, C_static, Y, mask,
            n_epochs=300, lambda_smooth=0.02, lr=0.01):
        """
        EM-style training (Study 1 일관성).
        
        Each epoch:
          E-step: forward_filter → Z_filtered (detached)
                  σ_Z affects Kalman gain → Z is noise-aware
          M-step: optimize outcome params given Z_filtered
                  transition params updated indirectly (next E-step)
        
        Also fits CovariateTransitionModel (Ridge) for g-formula simulation.
        """
        # 1. Fit covariate transition model (Ridge, non-parametric)
        self.cov_model.fit(L_dyn, S)

        # 2. EM optimization
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        patience = 30

        for epoch in tqdm(range(n_epochs), desc="Training (EM)", leave=False):
            # --- E-step: Forward filter (σ_Z participates via Kalman gain) ---
            with torch.no_grad():
                Z_filtered, Z_var, ll = self.forward_filter(
                    S, L_dyn, C_static, Y, mask
                )
                Z_est = Z_filtered.detach()

            # --- M-step: Optimize outcome/bin params given Z ---
            optimizer.zero_grad()
            loss = self.compute_loss(
                S, L_dyn, C_static, Y, mask, Z_est, lambda_smooth
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            # Early stopping
            curr_loss = loss.item()
            if curr_loss < best_loss - 1e-4:
                best_loss = curr_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return self

    # -----------------------------------------------------------------
    # G-Formula Simulation (Study 1 일관성)
    # -----------------------------------------------------------------
    def simulate_gformula(self, L_dyn_init, C_static, mp_scaled, n_time=30):
        """
        Parametric g-formula Monte Carlo simulation.
        
        Study 1의 simulate_gformula와 동일한 논리:
          1. Intervention: do(MP = mp_scaled) 고정
          2. Covariate evolution: L_t = Ridge(L_{t-1}, A_{t-1})
          3. Stochastic latent state: Z + N(0, σ_Z²)
          4. Cumulative survival 계산
        
        Args:
            L_dyn_init: (N, n_dyn) baseline dynamic covariates
            C_static:   (N, n_static) static covariates
            mp_scaled:  scalar, standardized log-MP intervention level
            n_time:     int, simulation horizon
        Returns:
            risk_trajectory: list of length n_time (cumulative risk at each t)
        """
        self.eval()
        N = C_static.shape[0]
        device = C_static.device

        Z = torch.zeros(N, 1, device=device)
        surv = torch.ones(N, 1, device=device)

        # Intervention: fixed MP level
        A_val = torch.full((N, 1), mp_scaled, device=device)
        A_bin = self._get_bin_onehot(A_val)
        S_val_np = np.full((N, 1), mp_scaled)

        # Dynamic covariates: will be evolved
        curr_L_np = L_dyn_init.cpu().numpy().copy()

        risk_traj = []

        with torch.no_grad():
            for t in range(n_time):
                L_tensor = torch.tensor(
                    curr_L_np, dtype=torch.float32, device=device
                )

                # Outcome probability
                prob_Y = self._outcome_probability(
                    Z, A_bin, L_tensor, C_static, t=t
                )
                surv = surv * (1 - prob_Y)
                risk_traj.append(1 - surv.mean().item())

                # Latent state transition (STOCHASTIC — Study 1 일관성)
                Z_mean, Z_var = self._latent_transition(
                    Z, A_val, L_tensor, C_static
                )
                Z = Z_mean + torch.randn_like(Z_mean) * torch.sqrt(Z_var)

                # Covariate evolution (ACTIVE — g-formula 핵심)
                if self.cov_model.is_fitted:
                    curr_L_np = self.cov_model.predict(curr_L_np, S_val_np)

        return risk_traj

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------
    def get_bin_effects(self, bin_edges_original=None):
        """
        Return bin coefficients for visualization.
        If bin_edges_original provided (unscaled MP values), maps back.
        """
        effects = self.beta_bins.detach().cpu().numpy()
        edges = self.bin_edges.cpu().numpy()
        centers = (edges[:-1] + edges[1:]) / 2
        return {'centers': centers, 'effects': effects}

    def get_parameters(self):
        """Return all learned parameters as dict."""
        return {
            'psi': self.psi.item(),
            'gamma_A': self.gamma_A.item(),
            'sigma_Z': self.sigma_Z.item(),
            'beta_0': self.beta_0.item(),
            'beta_Z': self.beta_Z.item(),
            'beta_time': self.beta_time.item(),
            'beta_bins': self.beta_bins.detach().cpu().numpy().tolist(),
        }


# =============================================================================
# Baseline: Standard G-Formula (No Latent State, for fair comparison)
# =============================================================================
class StandardGFormula(nn.Module):
    """
    Standard parametric g-formula WITHOUT latent state.
    Same binned structure + covariate evolution for fair comparison.
    The only difference from ContinuousHMM: no Z.
    """

    def __init__(self, n_dyn_covariates, n_static_covariates, n_bins=15):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer('bin_edges', torch.linspace(-2.5, 2.5, n_bins + 1))

        self.beta_0 = nn.Parameter(torch.tensor([-4.0]))
        self.beta_bins = nn.Parameter(torch.zeros(n_bins))
        self.beta_dyn = nn.Linear(n_dyn_covariates, 1, bias=False)
        self.beta_static = nn.Linear(n_static_covariates, 1, bias=False)
        self.beta_time = nn.Parameter(torch.tensor([0.03]))

        self.cov_model = CovariateTransitionModel(n_dyn_covariates)

    def _get_bin_onehot(self, S_cont):
        s = S_cont.contiguous().squeeze(-1)
        idx = torch.bucketize(s, self.bin_edges[1:-1])
        return F.one_hot(idx.clamp(0, self.n_bins - 1).long(), self.n_bins).float()

    def _outcome_probability(self, A_bin, L_dyn, C_static, t=0):
        logit = (self.beta_0
                 + (A_bin * self.beta_bins).sum(-1, keepdim=True)
                 + self.beta_dyn(L_dyn)
                 + self.beta_static(C_static)
                 + self.beta_time * (t / 30.0))
        return torch.sigmoid(logit.clamp(-20, 20))

    def fit(self, S, L_dyn, C_static, Y, mask,
            n_epochs=300, lambda_smooth=0.02, lr=0.01):
        self.cov_model.fit(L_dyn, S)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        S_bins = self._get_bin_onehot(S)

        for epoch in tqdm(range(n_epochs), desc="StdGF", leave=False):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=S.device)
            for t in range(S.shape[1]):
                A_bin_t = S_bins[:, t, :]
                L_t = L_dyn[:, t, :]
                Y_t = Y[:, t, :]
                m_t = mask[:, t].unsqueeze(1)
                prob = self._outcome_probability(A_bin_t, L_t, C_static, t=t)
                bce = (-Y_t * torch.log(prob + 1e-10)
                       - (1 - Y_t) * torch.log(1 - prob + 1e-10))
                loss = loss + (bce * m_t).sum() / (m_t.sum() + 1e-10)

            diff = self.beta_bins[1:] - self.beta_bins[:-1]
            loss = loss + lambda_smooth * torch.sum(diff ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        return self

    def simulate_gformula(self, L_dyn_init, C_static, mp_scaled, n_time=30):
        self.eval()
        N = C_static.shape[0]
        device = C_static.device
        surv = torch.ones(N, 1, device=device)
        A_bin = self._get_bin_onehot(
            torch.full((N, 1), mp_scaled, device=device)
        )
        S_np = np.full((N, 1), mp_scaled)
        curr_L_np = L_dyn_init.cpu().numpy().copy()
        risk_traj = []

        with torch.no_grad():
            for t in range(n_time):
                L_tensor = torch.tensor(curr_L_np, dtype=torch.float32, device=device)
                prob = self._outcome_probability(A_bin, L_tensor, C_static, t=t)
                surv *= (1 - prob)
                risk_traj.append(1 - surv.mean().item())
                if self.cov_model.is_fitted:
                    curr_L_np = self.cov_model.predict(curr_L_np, S_np)
        return risk_traj


# =============================================================================
# Baseline: Cox Proportional Hazards (Clinical Standard)
# =============================================================================
#
# WHY Cox PH as comparison?
#
# Cox PH는 ICU 임상 연구의 표준 분석 방법이다.
# 심사에서 "왜 Cox를 안 썼나?"는 반드시 나오는 질문이므로
# Cox PH를 비교 모델로 포함하고, 그 한계를 보여주는 것이 핵심.
#
# Cox PH의 한계 (= 제안 방법론의 존재 이유):
#   1. Time-varying confounding 미보정
#      → weaning bias: MP가 낮은 환자가 이미 호전/악화된 환자
#   2. 비선형 dose-response 포착 불가 (기본 선형 가정)
#   3. Latent state 없음 → unmeasured severity 미반영
#
# Implementation: lifelines CoxPHFitter
#   - Person-time format (환자-일 단위)
#   - Time-varying covariates 지원
#   - predict_survival_function으로 dose-response 추출
#
import pandas as pd

class CoxPHBaseline:
    """
    Time-varying Cox Proportional Hazards Model.
    
    Clinical standard comparison — does NOT correct for
    time-varying confounding (the key limitation this thesis addresses).
    
    Uses lifelines CoxPHFitter on person-time (patient-day) data.
    
    Requires: pip install lifelines
    """

    def __init__(self, penalizer=0.01):
        self.penalizer = penalizer
        self.model = None
        self.n_dyn = None
        self.n_static = None
        self.cov_cols = None

    def _tensor_to_person_time(self, S, L_dyn, C_static, Y, mask):
        """
        Convert tensor data → person-time DataFrame for lifelines.
        
        Each (patient i, day t) becomes one row:
          id, start, stop, event, mp, L0..Lk, C0..Cp
        """
        N, T, _ = S.shape
        self.n_dyn = L_dyn.shape[2]
        self.n_static = C_static.shape[1]

        rows = []
        for i in range(N):
            for t in range(T):
                if mask[i, t].item() == 0:
                    break
                row = {
                    'id': i,
                    'start': t,
                    'stop': t + 1,
                    'event': int(Y[i, t, 0].item()),
                    'mp': S[i, t, 0].item(),
                }
                for j in range(self.n_dyn):
                    row[f'L{j}'] = L_dyn[i, t, j].item()
                for j in range(self.n_static):
                    row[f'C{j}'] = C_static[i, j].item()
                rows.append(row)

        df = pd.DataFrame(rows)
        self.cov_cols = (
            ['mp']
            + [f'L{j}' for j in range(self.n_dyn)]
            + [f'C{j}' for j in range(self.n_static)]
        )
        return df

    def fit(self, S, L_dyn, C_static, Y, mask, **kwargs):
        """
        Fit time-varying Cox PH on person-time data.
        
        Same interface as ContinuousHMM.fit() for consistency in main.py.
        Extra kwargs are silently ignored (n_epochs, lambda_smooth, etc.)
        """
        try:
            from lifelines import CoxTimeVaryingFitter
        except ImportError:
            raise ImportError(
                "lifelines is required for Cox PH baseline.\n"
                "Install: pip install lifelines"
            )

        df = self._tensor_to_person_time(S, L_dyn, C_static, Y, mask)

        self.model = CoxTimeVaryingFitter(penalizer=self.penalizer)
        self.model.fit(
            df,
            id_col='id',
            event_col='event',
            start_col='start',
            stop_col='stop',
            show_progress=False,
        )

        # Store for dose-response prediction
        self._mean_covs = df[self.cov_cols].mean().to_dict()
        self._baseline_ch = self.model.baseline_cumulative_hazard_

        return self

    def predict_risk_at(self, mp_scaled, n_time=30):
        """
        Predict 30-day cumulative risk at a given MP level.
        
        Method:
          1. Compute partial hazard: exp(β_mp * mp + β_L * L̄ + β_C * C̄)
          2. Cumulative hazard: H(t) = H_0(t) * partial_hazard
          3. Survival: S(t) = exp(-H(t))
          4. Risk: 1 - S(t)
        
        Uses population-average covariates (marginal prediction).
        """
        # Build covariate vector: target MP + population-average L, C
        x = self._mean_covs.copy()
        x['mp'] = mp_scaled
        x_df = pd.DataFrame([x])[self.cov_cols]

        # Partial hazard = exp(X @ β)
        partial_hz = np.exp(
            (x_df.values @ self.model.params_.values).item()
        )

        # Baseline cumulative hazard at target time
        bch = self._baseline_ch
        max_t = bch.index.max()
        target_t = min(n_time, max_t)

        # Find closest available time point
        if target_t in bch.index:
            H0 = bch.loc[target_t].values[0]
        else:
            # Linear interpolation
            idx = bch.index.searchsorted(target_t)
            if idx >= len(bch):
                H0 = bch.iloc[-1].values[0]
            else:
                H0 = bch.iloc[idx].values[0]

        # Risk = 1 - exp(-H0 * partial_hazard)
        risk = 1 - np.exp(-H0 * partial_hz)
        return risk

    def simulate_gformula(self, L_dyn_init, C_static, mp_scaled, n_time=30):
        """
        Interface-compatible with ContinuousHMM.simulate_gformula().
        
        NOTE: This is NOT a g-formula simulation.
        Cox PH does not model covariate evolution under intervention.
        This returns the Cox-predicted risk trajectory (associational, not causal).
        The difference between this and the proposed model IS the thesis result.
        """
        risk_traj = []
        bch = self._baseline_ch

        # Build population-average covariate vector with target MP
        x = self._mean_covs.copy()
        x['mp'] = mp_scaled
        x_df = pd.DataFrame([x])[self.cov_cols]

        partial_hz = np.exp(
            (x_df.values @ self.model.params_.values).item()
        )

        for t in range(1, n_time + 1):
            # Cumulative baseline hazard at time t
            if t in bch.index:
                H0 = bch.loc[t].values[0]
            elif t > bch.index.max():
                H0 = bch.iloc[-1].values[0]
            else:
                idx = bch.index.searchsorted(t)
                H0 = bch.iloc[min(idx, len(bch) - 1)].values[0]

            surv = np.exp(-H0 * partial_hz)
            risk_traj.append(1 - surv)  # proportion (0-1), same as other models

        return risk_traj

    def get_summary(self):
        """Return Cox model summary for diagnostics."""
        if self.model is not None:
            return self.model.summary
        return None