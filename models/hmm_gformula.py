"""
models/hmm_gformula.py - Hidden Markov Model based g-formula (v3.2)

v3.2 Changes:
- [NEW] alpha_time: 학습 가능한 시간 효과 (흡연)
- [NEW] beta_time: 학습 가능한 시간 효과 (CVD)
- forward 단계에서 시간 t를 입력받아 반영

Models:
    Hidden State: Z_t = ψ*Z_{t-1} + γ_S*S_t + γ_C*C_t + γ_G*G + γ_L*L + γ_{GS}*(G×S_t) + ε
    Outcome: logit(Y_t) = β_0 + β_Z*Z_t + β_S*S_t + β_G*G + β_L*L + β_{GS}*(G×S_t) + β_time*t
    Exposure: logit(S_t) = α_0 + α_S*S_{t-1} + α_Z*Z_{t-1} + α_G*G + α_L*L + α_time*t
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_PARAMS, N_COVARIATES, GFORMULA_PARAMS


@dataclass
class HMMEstimates:
    """HMM 추정 결과"""
    Z_filtered: torch.Tensor
    Z_smoothed: torch.Tensor
    Z_variance: torch.Tensor
    log_likelihood: float
    converged: bool


class HiddenMarkovGFormula(nn.Module):
    """
    Hidden Markovian g-formula 모델 (v3.2)
    
    v3.2: 시간 효과(Aging) 파라미터 추가
    - alpha_time: 시간 경과에 따른 흡연 확률 변화
    - beta_time: 시간 경과에 따른 CVD 위험 변화
    """
    
    def __init__(
        self,
        n_covariates: int = N_COVARIATES,
        fit_interaction: bool = True,
        fit_pack_years: bool = True,
        use_hmm: bool = True,
        fit_time_effects: bool = True,  # [NEW] 시간 효과 학습 여부
    ):
        super().__init__()
        
        self.n_covariates = n_covariates
        self.fit_interaction = fit_interaction
        self.fit_pack_years = fit_pack_years
        self.use_hmm = use_hmm
        self.fit_time_effects = fit_time_effects
        
        # ===== Hidden State Transition Parameters =====
        self.psi = nn.Parameter(torch.tensor([0.5]))
        self.gamma_S = nn.Parameter(torch.tensor([0.1]))
        self.gamma_C = nn.Parameter(torch.tensor([0.05])) if fit_pack_years else None
        self.gamma_G = nn.Parameter(torch.tensor([0.1]))
        self.gamma_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        self.gamma_GC = nn.Parameter(torch.tensor([0.0])) if (fit_interaction and fit_pack_years) else None
        
        if n_covariates > 0:
            self.gamma_L = nn.Linear(n_covariates, 1, bias=False)
            nn.init.normal_(self.gamma_L.weight, mean=0.0, std=0.05)
        else:
            self.gamma_L = None
        
        self.log_sigma_Z = nn.Parameter(torch.tensor([-2.0]))
        
        # ===== Outcome Model Parameters =====
        self.beta_0 = nn.Parameter(torch.tensor([-5.0]))
        self.beta_Z = nn.Parameter(torch.tensor([0.3]))
        self.beta_S = nn.Parameter(torch.tensor([0.1]))
        self.beta_G = nn.Parameter(torch.tensor([0.1]))
        self.beta_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        
        if n_covariates > 0:
            self.beta_L = nn.Linear(n_covariates, 1, bias=False)
            nn.init.normal_(self.beta_L.weight, mean=0.0, std=0.1)
        else:
            self.beta_L = None
        
        # [NEW] 시간 효과 (Outcome)
        if fit_time_effects:
            self.beta_time = nn.Parameter(torch.tensor([0.03]))
        else:
            self.beta_time = None
        
        # ===== Exposure Model Parameters =====
        self.alpha_0 = nn.Parameter(torch.tensor([-2.0]))
        self.alpha_S = nn.Parameter(torch.tensor([1.5]))
        self.alpha_Z = nn.Parameter(torch.tensor([0.1]))
        self.alpha_G = nn.Parameter(torch.tensor([0.0]))
        
        if n_covariates > 0:
            self.alpha_L = nn.Linear(n_covariates, 1, bias=False)
            nn.init.normal_(self.alpha_L.weight, mean=0.0, std=0.1)
        else:
            self.alpha_L = None
        
        # [NEW] 시간 효과 (Exposure)
        if fit_time_effects:
            self.alpha_time = nn.Parameter(torch.tensor([-0.03]))
        else:
            self.alpha_time = None
    
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
        """Hidden state transition"""
        Z_mean = self.psi * Z_prev + self.gamma_S * S_curr + self.gamma_G * G
        
        if self.fit_pack_years and self.gamma_C is not None:
            Z_mean = Z_mean + self.gamma_C * C_curr
            
        if self.fit_interaction:
            if self.gamma_GS is not None:
                Z_mean = Z_mean + self.gamma_GS * (G * S_curr)
            if self.gamma_GC is not None and self.fit_pack_years:
                Z_mean = Z_mean + self.gamma_GC * (G * C_curr)
        
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
        t: int = 0,  # [NEW] 시간 입력
    ) -> torch.Tensor:
        """P(Y_t=1 | Z_t, S_t, G, L, t)"""
        logit = self.beta_0 + self.beta_Z * Z + self.beta_S * S + self.beta_G * G
        
        if self.fit_interaction and self.beta_GS is not None:
            logit = logit + self.beta_GS * (G * S)
        
        if self.beta_L is not None and L is not None:
            logit = logit + self.beta_L(L)
        
        # [NEW] 시간 효과
        if self.beta_time is not None:
            logit = logit + self.beta_time * t
        
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def _exposure_probability(
        self,
        S_prev: torch.Tensor,
        Z_prev: torch.Tensor,
        G: torch.Tensor,
        L: Optional[torch.Tensor] = None,
        t: int = 0,  # [NEW] 시간 입력
    ) -> torch.Tensor:
        """P(S_t=1 | S_{t-1}, Z_{t-1}, G, L, t)"""
        logit = self.alpha_0 + self.alpha_S * S_prev + self.alpha_Z * Z_prev + self.alpha_G * G
        
        if self.alpha_L is not None and L is not None:
            logit = logit + self.alpha_L(L)
        
        # [NEW] 시간 효과
        if self.alpha_time is not None:
            logit = logit + self.alpha_time * t
        
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def _create_at_risk_mask(self, Y: torch.Tensor) -> torch.Tensor:
        """중도절단 처리를 위한 at-risk mask"""
        n_samples, n_time, _ = Y.shape
        at_risk_mask = torch.ones_like(Y)
        
        for t in range(1, n_time):
            cumulative_event = (Y[:, :t, :].sum(dim=1, keepdim=True) > 0).float()
            at_risk_mask[:, t, :] = 1.0 - cumulative_event.squeeze(1)
        
        return at_risk_mask
    
    def forward_filter(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        L: Optional[torch.Tensor] = None,
    ) -> HMMEstimates:
        """Forward filtering"""
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
            
            Z_pred_mean, Z_pred_var_add = self._hidden_state_transition(
                Z_mean, S_t, C_t, G, L
            )
            Z_pred_var = (self.psi ** 2) * Z_var + Z_pred_var_add
            
            prob_Y = self._outcome_probability(Z_pred_mean, S_t, G, L, t=t)
            
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
        at_risk_mask: Optional[torch.Tensor] = None,
        Z_estimated: Optional[torch.Tensor] = None,
        use_survival_masking: bool = True,
    ) -> torch.Tensor:
        """Negative log-likelihood with censoring and time effects"""
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
        
        # At-risk mask
        if use_survival_masking:
            if at_risk_mask is None:
                at_risk_mask = self._create_at_risk_mask(Y)
        else:
            at_risk_mask = torch.ones_like(Y)
        
        # Compute masked loss
        total_loss = 0.0
        
        for t in range(n_time):
            prob_Y = self._outcome_probability(Z[:, t, :], S[:, t, :], G, L, t=t)
            Y_t = Y[:, t, :]
            mask_t = at_risk_mask[:, t, :]
            
            bce = -Y_t * torch.log(prob_Y + 1e-10) - (1 - Y_t) * torch.log(1 - prob_Y + 1e-10)
            masked_bce = bce * mask_t
            n_at_risk = mask_t.sum() + 1e-10
            
            total_loss = total_loss + masked_bce.sum() / n_at_risk
        
        reg_loss = 0.01 * self.sigma_Z ** 2
        
        return total_loss + reg_loss
    
    def fit(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        L: Optional[torch.Tensor] = None,
        at_risk_mask: Optional[torch.Tensor] = None,
        n_epochs: int = None,
        learning_rate: float = None,
        use_survival_masking: bool = True,
        verbose: bool = False,
    ) -> List[float]:
        """EM-style fitting"""
        n_epochs = n_epochs or TRAINING_PARAMS['n_epochs']
        lr = learning_rate or TRAINING_PARAMS['learning_rate']
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            if self.use_hmm:
                with torch.no_grad():
                    estimates = self.forward_filter(G, S, C, Y, L)
                    Z_est = estimates.Z_filtered.detach()
            else:
                Z_est = None
            
            loss = self.compute_loss(G, S, C, Y, L, at_risk_mask, Z_est, use_survival_masking)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and (epoch + 1) % 50 == 0:
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
        
        # [NEW] 시간 효과 파라미터
        if self.beta_time is not None:
            params['beta_time'] = self.beta_time.item()
        if self.alpha_time is not None:
            params['alpha_time'] = self.alpha_time.item()
        
        if self.gamma_L is not None:
            params['gamma_L'] = self.gamma_L.weight.data.squeeze().tolist()
        if self.beta_L is not None:
            params['beta_L'] = self.beta_L.weight.data.squeeze().tolist()
        if self.alpha_L is not None:
            params['alpha_L'] = self.alpha_L.weight.data.squeeze().tolist()
        
        return params
    
    def simulate_gformula(
        self,
        G: torch.Tensor,
        L: torch.Tensor,
        intervention: str = 'natural',
        n_time: int = 10,
        n_monte_carlo: int = None,
        quit_time: int = 0,
    ) -> Dict:
        """g-formula Monte Carlo 시뮬레이션 (시간 효과 포함)"""
        n_samples = G.shape[0]
        device = G.device
        n_mc = n_monte_carlo or GFORMULA_PARAMS.get('n_monte_carlo', 1000)
        
        G_exp = G.repeat(n_mc, 1)
        L_exp = L.repeat(n_mc, 1) if L is not None else None
        
        risk_trajectory = []
        cumulative_risk = torch.zeros(n_samples * n_mc, 1, device=device)
        survived = torch.ones(n_samples * n_mc, 1, device=device)
        
        Z_curr = torch.zeros(n_samples * n_mc, 1, device=device)
        S_prev = torch.zeros(n_samples * n_mc, 1, device=device)
        C_curr = torch.zeros(n_samples * n_mc, 1, device=device)
        
        with torch.no_grad():
            for t in range(n_time):
                # Intervention
                if intervention == 'always_smoke':
                    S_curr = torch.ones_like(S_prev)
                elif intervention == 'never_smoke':
                    S_curr = torch.zeros_like(S_prev)
                elif intervention.startswith('quit_at_t'):
                    if t >= quit_time:
                        S_curr = torch.zeros_like(S_prev)
                    else:
                        prob_S = self._exposure_probability(S_prev, Z_curr, G_exp, L_exp, t=t)
                        S_curr = torch.bernoulli(prob_S)
                else:  # natural
                    prob_S = self._exposure_probability(S_prev, Z_curr, G_exp, L_exp, t=t)
                    S_curr = torch.bernoulli(prob_S)
                
                C_curr = C_curr + S_curr
                
                Z_mean, Z_var = self._hidden_state_transition(Z_curr, S_curr, C_curr, G_exp, L_exp)
                Z_curr = Z_mean + torch.randn_like(Z_mean) * torch.sqrt(Z_var)
                
                prob_Y = self._outcome_probability(Z_curr, S_curr, G_exp, L_exp, t=t)
                
                cumulative_risk = cumulative_risk + survived * prob_Y
                survived = survived * (1 - prob_Y)
                
                risk_t = prob_Y.view(n_mc, n_samples, 1).mean(dim=0)
                risk_trajectory.append(risk_t.mean().item())
                
                S_prev = S_curr
        
        final_cumulative = cumulative_risk.view(n_mc, n_samples, 1).mean(dim=0)
        
        return {
            'risk_trajectory': risk_trajectory,
            'cumulative_risk': final_cumulative,
            'mean_cumulative_risk': final_cumulative.mean().item(),
            'cumulative_risk_by_subject': final_cumulative.squeeze(),
        }


def estimate_causal_effect(
    model: HiddenMarkovGFormula,
    G: torch.Tensor,
    L: torch.Tensor,
    n_time: int = 10,
    n_monte_carlo: int = None,
) -> Dict[str, float]:
    """g-formula를 이용한 인과 효과 추정"""
    results = {}
    n_mc = n_monte_carlo or GFORMULA_PARAMS.get('n_monte_carlo', 1000)
    
    for intervention in ['natural', 'always_smoke', 'never_smoke']:
        sim = model.simulate_gformula(G, L, intervention, n_time, n_mc)
        results[f'cumulative_risk_{intervention}'] = sim['mean_cumulative_risk']
        results[f'trajectory_{intervention}'] = sim['risk_trajectory']
    
    risk_always = results['cumulative_risk_always_smoke']
    risk_never = results['cumulative_risk_never_smoke']
    
    results['causal_risk_difference'] = risk_always - risk_never
    results['causal_risk_ratio'] = risk_always / (risk_never + 1e-10)
    
    return results


if __name__ == "__main__":
    print("Testing HMM-gFormula v3.2 (with Time Effects)...")
    
    import sys
    sys.path.insert(0, '..')
    from data_generator import generate_synthetic_data, validate_dgp
    
    data = generate_synthetic_data(n_samples=10000, n_time=10, seed=42)
    validate_dgp(data)
    
    model = HiddenMarkovGFormula(
        n_covariates=3, 
        fit_interaction=True, 
        fit_pack_years=True,
        fit_time_effects=True  # [NEW]
    )
    
    print("\nFitting model with time effects...")
    losses = model.fit(data.G, data.S, data.C, data.Y, L=data.L, n_epochs=100, verbose=True)
    
    print("\nEstimated parameters:")
    params = model.get_parameters()
    for k, v in params.items():
        if isinstance(v, list):
            print(f"  {k}: {[f'{x:.4f}' for x in v]}")
        else:
            print(f"  {k}: {v:.4f}")
    
    print("\nCausal effect estimation...")
    causal = estimate_causal_effect(model, data.G, data.L)
    print(f"  Risk Difference: {causal['causal_risk_difference']:.4f}")
    print(f"  Risk Ratio: {causal['causal_risk_ratio']:.4f}")