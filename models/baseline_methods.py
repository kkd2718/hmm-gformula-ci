"""
models/baseline_methods.py - Comparison Methods

비교 방법론:
1. Naive Logistic Regression (cross-sectional)
2. Pooled Logistic Regression (time-pooled)
3. Marginal Structural Model with IPTW
4. Time-varying coefficient model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import TRAINING_PARAMS
except ImportError:
    TRAINING_PARAMS = {'n_epochs': 200, 'learning_rate': 0.01}


@dataclass 
class BaselineResult:
    """결과 컨테이너"""
    coefficients: Dict[str, float]
    std_errors: Optional[Dict[str, float]] = None
    converged: bool = True


class NaiveLogistic(nn.Module):
    """
    Naive Logistic Regression
    
    마지막 시점의 결과만 사용하여 단면적 분석
    (HMM 무시, 시간 구조 무시)
    
    Model: logit(Y_T) = β_0 + β_S*S_T + β_G*G + β_{GS}*(G×S_T) + β_C*C_T
    """
    
    def __init__(self, fit_interaction: bool = True, fit_pack_years: bool = True):
        super().__init__()
        self.fit_interaction = fit_interaction
        self.fit_pack_years = fit_pack_years
        
        self.beta_0 = nn.Parameter(torch.tensor([-3.0]))
        self.beta_S = nn.Parameter(torch.tensor([0.1]))
        self.beta_G = nn.Parameter(torch.tensor([0.1]))
        self.beta_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        self.beta_C = nn.Parameter(torch.tensor([0.1])) if fit_pack_years else None
        
    def forward(self, G: torch.Tensor, S: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            G: (N, 1) PRS
            S: (N, 1) Final smoking status
            C: (N, 1) Final pack-years
        """
        logit = self.beta_0 + self.beta_S * S + self.beta_G * G
        
        if self.fit_interaction and self.beta_GS is not None:
            logit = logit + self.beta_GS * (G * S)
        
        if self.fit_pack_years and self.beta_C is not None:
            logit = logit + self.beta_C * C
            
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def fit(
        self, 
        G: torch.Tensor, 
        S: torch.Tensor, 
        C: torch.Tensor, 
        Y: torch.Tensor,
        n_epochs: int = None,
        lr: float = None,
    ) -> List[float]:
        """
        마지막 시점 데이터만 사용하여 fitting
        """
        n_epochs = n_epochs or TRAINING_PARAMS['n_epochs']
        lr = lr or TRAINING_PARAMS['learning_rate']
        
        # 마지막 시점 데이터
        S_final = S[:, -1, :]
        C_final = C[:, -1, :]
        Y_final = Y[:, -1, :]
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        losses = []
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = self.forward(G, S_final, C_final)
            loss = criterion(pred, Y_final)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses
    
    def get_parameters(self) -> Dict[str, float]:
        params = {
            'beta_0': self.beta_0.item(),
            'beta_S': self.beta_S.item(),
            'beta_G': self.beta_G.item(),
        }
        if self.fit_interaction and self.beta_GS is not None:
            params['beta_GS'] = self.beta_GS.item()
        if self.fit_pack_years and self.beta_C is not None:
            params['beta_C'] = self.beta_C.item()
        return params


class PooledLogistic(nn.Module):
    """
    Pooled Logistic Regression (Discrete-time survival model)
    
    모든 시점의 데이터를 pooling하여 분석
    
    Model: logit(Y_t) = β_0 + β_t*t + β_S*S_t + β_G*G + β_{GS}*(G×S_t) + β_C*C_t
    """
    
    def __init__(self, fit_interaction: bool = True, fit_pack_years: bool = True):
        super().__init__()
        self.fit_interaction = fit_interaction
        self.fit_pack_years = fit_pack_years
        
        self.beta_0 = nn.Parameter(torch.tensor([-3.0]))
        self.beta_t = nn.Parameter(torch.tensor([0.05]))  # Time trend
        self.beta_S = nn.Parameter(torch.tensor([0.1]))
        self.beta_G = nn.Parameter(torch.tensor([0.1]))
        self.beta_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        self.beta_C = nn.Parameter(torch.tensor([0.1])) if fit_pack_years else None
        
    def forward(
        self, 
        G: torch.Tensor, 
        S: torch.Tensor, 
        C: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            G: (N*T, 1) PRS (expanded)
            S: (N*T, 1) Smoking status
            C: (N*T, 1) Pack-years
            t: (N*T, 1) Time indicator
        """
        logit = self.beta_0 + self.beta_t * t + self.beta_S * S + self.beta_G * G
        
        if self.fit_interaction and self.beta_GS is not None:
            logit = logit + self.beta_GS * (G * S)
            
        if self.fit_pack_years and self.beta_C is not None:
            logit = logit + self.beta_C * C
            
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def fit(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        n_epochs: int = None,
        lr: float = None,
    ) -> List[float]:
        """
        전체 panel 데이터를 long format으로 변환하여 fitting
        """
        n_epochs = n_epochs or TRAINING_PARAMS['n_epochs']
        lr = lr or TRAINING_PARAMS['learning_rate']
        
        n_samples, n_time, _ = S.shape
        
        # Reshape to long format
        G_long = G.repeat(1, n_time).view(-1, 1)
        S_long = S.view(-1, 1)
        C_long = C.view(-1, 1)
        Y_long = Y.view(-1, 1)
        t_long = torch.arange(n_time).float().repeat(n_samples).view(-1, 1)
        t_long = t_long / n_time  # Normalize
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        losses = []
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = self.forward(G_long, S_long, C_long, t_long)
            loss = criterion(pred, Y_long)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses
    
    def get_parameters(self) -> Dict[str, float]:
        params = {
            'beta_0': self.beta_0.item(),
            'beta_t': self.beta_t.item(),
            'beta_S': self.beta_S.item(),
            'beta_G': self.beta_G.item(),
        }
        if self.fit_interaction and self.beta_GS is not None:
            params['beta_GS'] = self.beta_GS.item()
        if self.fit_pack_years and self.beta_C is not None:
            params['beta_C'] = self.beta_C.item()
        return params


class MSM_IPTW(nn.Module):
    """
    Marginal Structural Model with Inverse Probability of Treatment Weighting
    
    1단계: Propensity score 모델 fitting
    2단계: IPTW를 이용한 weighted logistic regression
    
    PS Model: logit(S_t) = α_0 + α_S*S_{t-1} + α_G*G
    Outcome: logit(Y_t) = β_0 + β_S*S_t + β_G*G + β_{GS}*(G×S_t) [weighted]
    """
    
    def __init__(self, fit_interaction: bool = True, stabilized: bool = True):
        super().__init__()
        self.fit_interaction = fit_interaction
        self.stabilized = stabilized
        
        # Propensity score model
        self.alpha_0 = nn.Parameter(torch.tensor([-1.0]))
        self.alpha_S = nn.Parameter(torch.tensor([1.0]))
        self.alpha_G = nn.Parameter(torch.tensor([0.1]))
        
        # Outcome model (MSM)
        self.beta_0 = nn.Parameter(torch.tensor([-3.0]))
        self.beta_S = nn.Parameter(torch.tensor([0.1]))
        self.beta_G = nn.Parameter(torch.tensor([0.1]))
        self.beta_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        
    def propensity_score(
        self, 
        S_prev: torch.Tensor, 
        G: torch.Tensor
    ) -> torch.Tensor:
        """P(S_t=1 | S_{t-1}, G)"""
        logit = self.alpha_0 + self.alpha_S * S_prev + self.alpha_G * G
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def compute_weights(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute stabilized IPTW weights
        
        Args:
            G: (N, 1)
            S: (N, T, 1)
            
        Returns:
            weights: (N, T, 1)
        """
        n_samples, n_time, _ = S.shape
        
        weights = torch.ones(n_samples, n_time, 1)
        S_prev = torch.zeros(n_samples, 1)
        
        # Marginal probability for stabilization
        if self.stabilized:
            marginal_p = S.mean().item()
        
        for t in range(n_time):
            S_t = S[:, t, :]
            ps = self.propensity_score(S_prev, G)
            
            # P(S_t | history)
            prob_obs = S_t * ps + (1 - S_t) * (1 - ps)
            
            if self.stabilized:
                # Stabilized weight: P(S_t) / P(S_t | history)
                prob_marg = S_t * marginal_p + (1 - S_t) * (1 - marginal_p)
                w_t = prob_marg / (prob_obs + 1e-10)
            else:
                w_t = 1.0 / (prob_obs + 1e-10)
            
            weights[:, t, :] = w_t
            S_prev = S_t
        
        # Cumulative weights (product over time)
        cum_weights = weights.cumprod(dim=1)
        
        # Truncate extreme weights (1st and 99th percentile)
        flat_weights = cum_weights.view(-1)
        lower = torch.quantile(flat_weights, 0.01)
        upper = torch.quantile(flat_weights, 0.99)
        cum_weights = cum_weights.clamp(lower, upper)
        
        return cum_weights
    
    def outcome_model(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
    ) -> torch.Tensor:
        """MSM outcome model"""
        logit = self.beta_0 + self.beta_S * S + self.beta_G * G
        
        if self.fit_interaction and self.beta_GS is not None:
            logit = logit + self.beta_GS * (G * S)
            
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def fit(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,  # Not used but kept for interface consistency
        Y: torch.Tensor,
        n_epochs: int = None,
        lr: float = None,
    ) -> List[float]:
        """
        2-stage fitting
        """
        n_epochs = n_epochs or TRAINING_PARAMS['n_epochs']
        lr = lr or TRAINING_PARAMS['learning_rate']
        
        n_samples, n_time, _ = S.shape
        
        # Stage 1: Fit propensity score model
        ps_params = [self.alpha_0, self.alpha_S, self.alpha_G]
        optimizer_ps = optim.Adam(ps_params, lr=lr)
        
        for _ in range(n_epochs // 2):
            optimizer_ps.zero_grad()
            loss = 0.0
            S_prev = torch.zeros(n_samples, 1)
            
            for t in range(n_time):
                S_t = S[:, t, :]
                ps = self.propensity_score(S_prev, G)
                bce = -S_t * torch.log(ps + 1e-10) - (1 - S_t) * torch.log(1 - ps + 1e-10)
                loss = loss + bce.mean()
                S_prev = S_t
                
            loss.backward()
            optimizer_ps.step()
        
        # Stage 2: Fit weighted outcome model
        with torch.no_grad():
            weights = self.compute_weights(G, S)
        
        outcome_params = [self.beta_0, self.beta_S, self.beta_G]
        if self.beta_GS is not None:
            outcome_params.append(self.beta_GS)
        optimizer_out = optim.Adam(outcome_params, lr=lr)
        
        losses = []
        
        # Convert to long format
        G_long = G.repeat(1, n_time).view(-1, 1)
        S_long = S.view(-1, 1)
        Y_long = Y.view(-1, 1)
        W_long = weights.view(-1, 1).detach()
        
        for _ in range(n_epochs // 2):
            optimizer_out.zero_grad()
            
            pred = self.outcome_model(G_long, S_long)
            
            # Weighted BCE
            bce = -Y_long * torch.log(pred + 1e-10) - (1 - Y_long) * torch.log(1 - pred + 1e-10)
            weighted_loss = (bce * W_long).mean()
            
            weighted_loss.backward()
            optimizer_out.step()
            losses.append(weighted_loss.item())
            
        return losses
    
    def get_parameters(self) -> Dict[str, float]:
        params = {
            'beta_0': self.beta_0.item(),
            'beta_S': self.beta_S.item(),
            'beta_G': self.beta_G.item(),
            'alpha_0': self.alpha_0.item(),
            'alpha_S': self.alpha_S.item(),
            'alpha_G': self.alpha_G.item(),
        }
        if self.fit_interaction and self.beta_GS is not None:
            params['beta_GS'] = self.beta_GS.item()
        return params


class TimeVaryingCoefficient(nn.Module):
    """
    Time-varying coefficient model
    
    계수가 시간에 따라 변화하는 모델 (simplified version)
    
    logit(Y_t) = β_0(t) + β_S(t)*S_t + β_G*G + β_{GS}(t)*(G×S_t)
    
    where β(t) = β + δ*t (linear time trend in coefficients)
    """
    
    def __init__(self, fit_interaction: bool = True):
        super().__init__()
        self.fit_interaction = fit_interaction
        
        # Base coefficients
        self.beta_0 = nn.Parameter(torch.tensor([-3.0]))
        self.beta_S = nn.Parameter(torch.tensor([0.1]))
        self.beta_G = nn.Parameter(torch.tensor([0.1]))
        self.beta_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        
        # Time trend coefficients
        self.delta_0 = nn.Parameter(torch.tensor([0.0]))
        self.delta_S = nn.Parameter(torch.tensor([0.0]))
        self.delta_GS = nn.Parameter(torch.tensor([0.0])) if fit_interaction else None
        
    def forward(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            G, S: (N, 1)
            t: (N, 1) normalized time
        """
        beta_0_t = self.beta_0 + self.delta_0 * t
        beta_S_t = self.beta_S + self.delta_S * t
        
        logit = beta_0_t + beta_S_t * S + self.beta_G * G
        
        if self.fit_interaction:
            beta_GS_t = self.beta_GS + self.delta_GS * t
            logit = logit + beta_GS_t * (G * S)
            
        return torch.sigmoid(logit.clamp(-20, 20))
    
    def fit(
        self,
        G: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        n_epochs: int = None,
        lr: float = None,
    ) -> List[float]:
        n_epochs = n_epochs or TRAINING_PARAMS['n_epochs']
        lr = lr or TRAINING_PARAMS['learning_rate']
        
        n_samples, n_time, _ = S.shape
        
        # Long format
        G_long = G.repeat(1, n_time).view(-1, 1)
        S_long = S.view(-1, 1)
        Y_long = Y.view(-1, 1)
        t_long = torch.arange(n_time).float().repeat(n_samples).view(-1, 1) / n_time
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = self.forward(G_long, S_long, t_long)
            loss = nn.functional.binary_cross_entropy(pred, Y_long)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses
    
    def get_parameters(self) -> Dict[str, float]:
        params = {
            'beta_0': self.beta_0.item(),
            'beta_S': self.beta_S.item(),
            'beta_G': self.beta_G.item(),
            'delta_0': self.delta_0.item(),
            'delta_S': self.delta_S.item(),
        }
        if self.fit_interaction:
            params['beta_GS'] = self.beta_GS.item()
            params['delta_GS'] = self.delta_GS.item()
        return params


# =============================================================================
# Factory function
# =============================================================================

def get_baseline_model(method: str, **kwargs) -> nn.Module:
    """
    방법론 이름으로 모델 인스턴스 반환
    """
    models = {
        'naive_logistic': NaiveLogistic,
        'pooled_logistic': PooledLogistic,
        'msm_iptw': MSM_IPTW,
        'time_varying': TimeVaryingCoefficient,
    }
    
    if method.lower() not in models:
        raise ValueError(f"Unknown method: {method}. Available: {list(models.keys())}")
    
    return models[method.lower()](**kwargs)


if __name__ == "__main__":
    print("Testing Baseline Methods...")
    
    import sys
    sys.path.insert(0, '..')
    from data_generator import generate_synthetic_data
    
    data = generate_synthetic_data(n_samples=3000, n_time=10, seed=42)
    
    methods = ['naive_logistic', 'pooled_logistic', 'msm_iptw', 'time_varying']
    
    for method_name in methods:
        print(f"\n{'='*50}")
        print(f"Testing: {method_name}")
        print('='*50)
        
        model = get_baseline_model(method_name, fit_interaction=True)
        losses = model.fit(data.G, data.S, data.C, data.Y, n_epochs=100)
        
        print(f"Final loss: {losses[-1]:.4f}")
        print("Parameters:")
        for k, v in model.get_parameters().items():
            print(f"  {k}: {v:.4f}")