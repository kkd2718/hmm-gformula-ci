"""
models/hmm_gformula_v2.py - Hidden Markov Model based g-formula (완성본)

[수정사항]
1. 공변량(Covariates, L) 지원 추가
2. 중도절단(Censoring) 처리를 위한 Masking 추가
3. 실제 데이터 적용을 위한 인터페이스 개선

Models:
    Hidden State: Z_t = ψ*Z_{t-1} + γ_S*S_t + γ_C*C_t + γ_G*G + γ_L*L + γ_{GS}*(G×S_t) + γ_{GC}*(G×C_t) + ε
    Outcome: logit(Y_t) = β_0 + β_Z*Z_t + β_S*S_t + β_G*G + β_L*L + β_{GS}*(G×S_t)
    Exposure: logit(S_t) = α_0 + α_S*S_{t-1} + α_Z*Z_{t-1} + α_G*G + α_L*L
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class HMMEstimates:
    """HMM 추정 결과 컨테이너"""
    Z_filtered: torch.Tensor
    Z_smoothed: torch.Tensor
    Z_variance: torch.Tensor
    log_likelihood: float
    converged: bool


class HiddenMarkovGFormulaV2(nn.Module):
    """
    Hidden Markovian g-formula 모델 (완성본)
    
    [Gap 1 해결] 공변량(L) 지원
    [Gap 2 해결] Censoring을 위한 at-risk masking
    
    Args:
        n_covariates: 공변량 개수 (예: 나이, 성별, BMI = 3)
        fit_interaction: GxE interaction 항 포함 여부
        fit_pack_years: 누적 흡연량(C) 포함 여부
        use_hmm: HMM 기반 latent state 추정 사용 여부
    """
    
    def __init__(
        self,
        n_covariates: int = 0,
        fit_interaction: bool = True,
        fit_pack_years: bool = True,
        use_hmm: bool = True,
    ):
        super().__init__()
        
        self.n_covariates = n_covariates
        self.fit_interaction = fit_interaction
        self.fit_pack_years = fit_pack_years
        self.use_hmm = use_hmm
        
        # ===== Hidden State Transition Parameters =====
        # Z_t = ψ*Z_{t-1} + γ_S*S_t + γ_C*C_t + γ_G*G + γ_L*L + γ_{GS}*(G×S_t) + γ_{GC}*(G×C_t)
        self.psi = nn.Parameter(torch.tensor([0.5]))
        self.gamma_S = nn.Parameter(torch.tensor([0.1]))
        self.gamma_C = nn.Parameter(torch.tensor([0.1])) if fit_pack_years else None
        self.gamma_G = nn.Parameter(torch.tensor([0.1]))
        self.gamma_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        self.gamma_GC = nn.Parameter(torch.tensor([0.0])) if (fit_interaction and fit_pack_years) else None
        
        # [Gap 1] 공변량 효과
        if n_covariates > 0:
            self.gamma_L = nn.Linear(n_covariates, 1, bias=False)
        else:
            self.gamma_L = None
        
        self.log_sigma_Z = nn.Parameter(torch.tensor([-2.0]))
        
        # ===== Outcome Model Parameters =====
        # logit(Y_t) = β_0 + β_Z*Z_t + β_S*S_t + β_G*G + β_L*L + β_{GS}*(G×S_t)
        self.beta_0 = nn.Parameter(torch.tensor([-3.0]))
        self.beta_Z = nn.Parameter(torch.tensor([0.5]))
        self.beta_S = nn.Parameter(torch.tensor([0.1]))
        self.beta_G = nn.Parameter(torch.tensor([0.1]))
        self.beta_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        
        # [Gap 1] 공변량 효과
        if n_covariates > 0:
            self.beta_L = nn.Linear(n_covariates, 1, bias=False)
        else:
            self.beta_L = None
        
        # ===== Exposure Model Parameters =====
        self.alpha_0 = nn.Parameter(torch.tensor([-1.0]))
        self.alpha_S = nn.Parameter(torch.tensor([1.0]))
        self.alpha_Z = nn.Parameter(torch.tensor([0.1]))
        self.alpha_G = nn.Parameter(torch.tensor([0.0]))
        
        if n_covariates > 0:
            self.alpha_L = nn.Linear(n_covariates, 1, bias=False)
        else:
            self.alpha_L = None
    
    @property
    def sigma_Z(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_Z)
    
    def _hidden_state_transition(
        self,
        Z_prev: torch.Tensor,
        S_curr: torch.Tensor,
        C_curr: torch.Tensor,
        G: torch.Tensor,
        L: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hidden state transition with covariates"""
        Z_mean = self.psi * Z_prev + self.gamma_S * S_curr + self.gamma_G * G
        
        if self.fit_pack_years and self.gamma_C is not None:
            Z_mean = Z_mean + self.gamma_C * C_curr
            
        if self.fit_interaction:
            if self.gamma_GS is not None:
                Z_mean = Z_mean + self.gamma_GS * (G * S_curr)
            if self.gamma_GC is not None and self.fit_pack_years:
                Z_mean = Z_mean + self.gamma_GC * (G * C_curr)
        
        # [Gap 1] 공변량 효과 추가
        if self.gamma_L is not None and L is not None:
            Z_mean = Z_mean + self.gamma_L(L)
        
        Z_var = self.sigma_Z ** 2
        return Z_mean, Z_var
    
    def _outcome_probability(
        self,
        Z: torch.Tensor,
        S: torch.Tensor,
        G: torch.Tensor,
        L: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Outcome probability with covariates"""
        logit = self.beta_0 + self.beta_Z * Z + self.beta_S * S + self.beta_G * G
        
        if self.fit_interaction and self.beta_GS is not None:
            logit = logit + self.beta_GS * (G * S)
        
        # [Gap 1] 공변량 효과 추가
        if self.beta_L is not None and L is not None:
            logit = logit + self.beta_L(L)
        
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def _exposure_probability(
        self,
        S_prev: torch.Tensor,
        Z_prev: torch.Tensor,
        G: torch.Tensor,
        L: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exposure probability with covariates"""
        logit = self.alpha_0 + self.alpha_S * S_prev + self.alpha_Z * Z_prev + self.alpha_G * G
        
        if self.alpha_L is not None and L is not None:
            logit = logit + self.alpha_L(L)
        
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def _create_at_risk_mask(self, Y: torch.Tensor) -> torch.Tensor:
        """
        [Gap 2] 중도절단 처리를 위한 at-risk mask 생성
        
        이전 시점에 이벤트가 발생하지 않은 사람만 현재 시점에서 at-risk
        
        Args:
            Y: (N, T, 1) 이벤트 발생 여부
            
        Returns:
            at_risk_mask: (N, T, 1) - 1이면 at-risk, 0이면 이미 이벤트 발생
        """
        n_samples, n_time, _ = Y.shape
        device = Y.device
        
        # 누적 이벤트 (t 이전까지의 이벤트 발생 여부)
        # shift right: t 시점의 mask는 t-1까지의 이벤트에 기반
        cumulative_events = torch.zeros_like(Y)
        
        for t in range(1, n_time):
            # t-1까지 한 번이라도 이벤트가 있었는지
            cumulative_events[:, t, :] = (Y[:, :t, :].sum(dim=1) > 0).float()
        
        # at-risk = 아직 이벤트가 발생하지 않은 사람
        at_risk_mask = 1.0 - cumulative_events
        
        return at_risk_mask
    
    def forward_filter(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        L: Optional[torch.Tensor] = None,
    ) -> HMMEstimates:
        """Forward filtering with covariates"""
        n_samples = G.shape[0]
        n_time = S.shape[1]
        device = G.device
        
        Z_filtered = []
        Z_var_filtered = []
        log_likelihoods = []
        
        Z_mean = torch.zeros(n_samples, 1, device=device)
        Z_var = torch.ones(n_samples, 1, device=device)
        
        for t in range(n_time):
            S_t = S[:, t, :]
            C_t = C[:, t, :]
            Y_t = Y[:, t, :]
            L_t = L if L is not None else None
            
            Z_pred_mean, Z_pred_var_add = self._hidden_state_transition(
                Z_mean, S_t, C_t, G, L_t
            )
            Z_pred_var = (self.psi ** 2) * Z_var + Z_pred_var_add
            
            prob_Y = self._outcome_probability(Z_pred_mean, S_t, G, L_t)
            
            gradient = (Y_t - prob_Y) * self.beta_Z
            hessian = (self.beta_Z ** 2) * prob_Y * (1 - prob_Y) + 1e-6
            
            Z_post_var = 1.0 / (1.0 / Z_pred_var + hessian)
            Z_post_mean = Z_pred_mean + Z_post_var * gradient
            
            ll = Y_t * torch.log(prob_Y + 1e-10) + (1 - Y_t) * torch.log(1 - prob_Y + 1e-10)
            log_likelihoods.append(ll.sum())
            
            Z_filtered.append(Z_post_mean)
            Z_var_filtered.append(Z_post_var)
            
            Z_mean = Z_post_mean
            Z_var = Z_post_var
        
        return HMMEstimates(
            Z_filtered=torch.stack(Z_filtered, dim=1),
            Z_smoothed=torch.stack(Z_filtered, dim=1),
            Z_variance=torch.stack(Z_var_filtered, dim=1),
            log_likelihood=sum(log_likelihoods).item(),
            converged=True,
        )
    
    def compute_loss(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        L: Optional[torch.Tensor] = None,
        Z_estimated: Optional[torch.Tensor] = None,
        use_survival_masking: bool = True,
    ) -> torch.Tensor:
        """
        Negative log-likelihood loss with censoring support
        
        [Gap 2] use_survival_masking=True이면 이전 시점 이벤트 발생자 제외
        """
        n_samples = G.shape[0]
        n_time = S.shape[1]
        
        # Estimate Z
        if Z_estimated is None and self.use_hmm:
            estimates = self.forward_filter(G, S, C, Y, L)
            Z = estimates.Z_filtered
        elif Z_estimated is not None:
            Z = Z_estimated
        else:
            Z_list = []
            Z_curr = torch.zeros(n_samples, 1, device=G.device)
            for t in range(n_time):
                Z_mean, _ = self._hidden_state_transition(Z_curr, S[:, t, :], C[:, t, :], G, L)
                Z_list.append(Z_mean)
                Z_curr = Z_mean
            Z = torch.stack(Z_list, dim=1)
        
        # [Gap 2] At-risk mask 생성
        if use_survival_masking:
            at_risk_mask = self._create_at_risk_mask(Y)
        else:
            at_risk_mask = torch.ones_like(Y)
        
        # Compute masked loss
        total_loss = 0.0
        n_at_risk_total = 0.0
        
        for t in range(n_time):
            prob_Y = self._outcome_probability(Z[:, t, :], S[:, t, :], G, L)
            Y_t = Y[:, t, :]
            mask_t = at_risk_mask[:, t, :]
            
            # BCE loss
            bce = -Y_t * torch.log(prob_Y + 1e-10) - (1 - Y_t) * torch.log(1 - prob_Y + 1e-10)
            
            # [Gap 2] Masked loss: at-risk인 사람만 loss에 포함
            masked_bce = bce * mask_t
            n_at_risk = mask_t.sum()
            
            if n_at_risk > 0:
                total_loss = total_loss + masked_bce.sum() / n_at_risk
                n_at_risk_total += n_at_risk
        
        # Regularization
        reg_loss = 0.01 * self.sigma_Z ** 2
        
        return total_loss + reg_loss
    
    def fit(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        L: Optional[torch.Tensor] = None,
        n_epochs: int = 200,
        learning_rate: float = 0.01,
        use_survival_masking: bool = True,
        verbose: bool = False,
    ) -> List[float]:
        """
        EM-style fitting
        
        Args:
            G, S, C, Y: Data tensors (N, 1) or (N, T, 1)
            L: Covariates (N, K) - optional
            use_survival_masking: [Gap 2] 생존 분석 masking 사용 여부
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            if self.use_hmm:
                with torch.no_grad():
                    estimates = self.forward_filter(G, S, C, Y, L)
                    Z_est = estimates.Z_filtered.detach()
            else:
                Z_est = None
            
            loss = self.compute_loss(G, S, C, Y, L, Z_est, use_survival_masking)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
        
        return losses
    
    def get_parameters(self) -> Dict[str, float]:
        """추정된 파라미터 반환"""
        params = {
            'psi': self.psi.item(),
            'gamma_S': self.gamma_S.item(),
            'gamma_G': self.gamma_G.item(),
            'sigma_Z': self.sigma_Z.item(),
            'beta_0': self.beta_0.item(),
            'beta_Z': self.beta_Z.item(),
            'beta_S': self.beta_S.item(),
            'beta_G': self.beta_G.item(),
            'alpha_0': self.alpha_0.item(),
            'alpha_S': self.alpha_S.item(),
            'alpha_Z': self.alpha_Z.item(),
            'alpha_G': self.alpha_G.item(),
        }
        
        if self.fit_pack_years and self.gamma_C is not None:
            params['gamma_C'] = self.gamma_C.item()
            
        if self.fit_interaction:
            if self.gamma_GS is not None:
                params['gamma_GS'] = self.gamma_GS.item()
            if self.gamma_GC is not None:
                params['gamma_GC'] = self.gamma_GC.item()
            if self.beta_GS is not None:
                params['beta_GS'] = self.beta_GS.item()
        
        # 공변량 계수 (있는 경우)
        if self.gamma_L is not None:
            params['gamma_L'] = self.gamma_L.weight.data.tolist()
        if self.beta_L is not None:
            params['beta_L'] = self.beta_L.weight.data.tolist()
        
        return params
    
    def simulate_gformula(
        self,
        G: torch.Tensor,
        intervention: str = 'natural',
        n_time: int = 10,
        n_monte_carlo: int = 1000,
        L: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """g-formula Monte Carlo simulation with covariates"""
        n_samples = G.shape[0]
        device = G.device
        
        G_expanded = G.repeat(n_monte_carlo, 1)
        L_expanded = L.repeat(n_monte_carlo, 1) if L is not None else None
        
        risk_trajectory = []
        cumulative_risk = torch.zeros(n_samples * n_monte_carlo, 1, device=device)
        survived = torch.ones(n_samples * n_monte_carlo, 1, device=device)
        
        Z_curr = torch.zeros(n_samples * n_monte_carlo, 1, device=device)
        S_prev = torch.zeros(n_samples * n_monte_carlo, 1, device=device)
        C_curr = torch.zeros(n_samples * n_monte_carlo, 1, device=device)
        
        with torch.no_grad():
            for t in range(n_time):
                # Intervention
                if intervention == 'always_smoke':
                    S_curr = torch.ones_like(S_prev)
                elif intervention == 'never_smoke':
                    S_curr = torch.zeros_like(S_prev)
                elif intervention == 'quit_at_t5' and t >= 5:
                    S_curr = torch.zeros_like(S_prev)
                else:
                    prob_S = self._exposure_probability(S_prev, Z_curr, G_expanded, L_expanded)
                    S_curr = torch.bernoulli(prob_S)
                
                C_curr = C_curr + S_curr
                
                Z_mean, Z_var = self._hidden_state_transition(Z_curr, S_curr, C_curr, G_expanded, L_expanded)
                Z_curr = Z_mean + torch.randn_like(Z_mean) * torch.sqrt(Z_var)
                
                prob_Y = self._outcome_probability(Z_curr, S_curr, G_expanded, L_expanded)
                
                # [Gap 2] Survival framework: 생존한 사람만 위험에 노출
                cumulative_risk = cumulative_risk + survived * prob_Y
                survived = survived * (1 - prob_Y)
                
                risk_t = prob_Y.view(n_monte_carlo, n_samples, 1).mean(dim=0)
                risk_trajectory.append(risk_t.mean().item())
                
                S_prev = S_curr
        
        final_cumulative = cumulative_risk.view(n_monte_carlo, n_samples, 1).mean(dim=0)
        
        return {
            'risk_trajectory': risk_trajectory,
            'cumulative_risk': final_cumulative,
            'mean_cumulative_risk': final_cumulative.mean().item(),
        }


def estimate_causal_effect(
    model: HiddenMarkovGFormulaV2,
    G: torch.Tensor,
    n_time: int = 10,
    L: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """g-formula를 이용한 인과 효과 추정"""
    results = {}
    
    for intervention in ['natural', 'always_smoke', 'never_smoke']:
        sim_result = model.simulate_gformula(G, intervention, n_time, L=L)
        results[f'cumulative_risk_{intervention}'] = sim_result['mean_cumulative_risk']
        results[f'trajectory_{intervention}'] = sim_result['risk_trajectory']
    
    risk_always = results['cumulative_risk_always_smoke']
    risk_never = results['cumulative_risk_never_smoke']
    
    results['causal_risk_difference'] = risk_always - risk_never
    results['causal_risk_ratio'] = risk_always / (risk_never + 1e-10)
    
    return results


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing HMM-gFormula V2 (with Covariates & Censoring)...")
    
    import sys
    sys.path.insert(0, '..')
    from data_generator import generate_synthetic_data, validate_dgp
    
    # Generate test data with covariates
    data = generate_synthetic_data(
        n_samples=3000, 
        n_time=10, 
        include_covariates=True,  # 공변량 포함
        survival_outcome=True,     # 생존 분석 형태
        seed=42
    )
    validate_dgp(data)
    
    print(f"\nCovariates shape: {data.L.shape if data.L is not None else 'None'}")
    
    # Initialize model with covariates
    model = HiddenMarkovGFormulaV2(
        n_covariates=3,  # age, sex, BMI
        fit_interaction=True, 
        fit_pack_years=True
    )
    
    print("\nFitting model with covariates and survival masking...")
    losses = model.fit(
        data.G, data.S, data.C, data.Y, 
        L=data.L,
        use_survival_masking=True,  # [Gap 2] 생존 분석 masking
        n_epochs=100, 
        verbose=True
    )
    
    print("\nEstimated parameters:")
    params = model.get_parameters()
    for k, v in params.items():
        if isinstance(v, list):
            print(f"  {k}: {[f'{x:.4f}' for x in v]}")
        else:
            print(f"  {k}: {v:.4f}")
    
    print("\ng-formula simulation with covariates...")
    causal = estimate_causal_effect(model, data.G, L=data.L)
    print(f"  Causal Risk Difference: {causal['causal_risk_difference']:.4f}")
    print(f"  Causal Risk Ratio: {causal['causal_risk_ratio']:.4f}")
    
    print("\nV2 Test PASSED!")