import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.linear_model import Ridge
import numpy as np

# --- [Shared] Covariate Dynamics (Ridge) ---
class CovariateTransitionModel:
    def __init__(self, n_covariates):
        self.models = [Ridge(alpha=1.0) for _ in range(n_covariates)]
        self.is_fitted = False

    def fit(self, L, S):
        L_np = L.cpu().numpy(); S_np = S.cpu().numpy()
        N, T, F = L_np.shape
        X = np.hstack([L_np[:, :-1, :].reshape(-1, F), S_np[:, :-1, :].reshape(-1, 1)])
        Y = L_np[:, 1:, :].reshape(-1, F)
        for i, model in enumerate(self.models):
            model.fit(X, Y[:, i])
        self.is_fitted = True

    def predict(self, L_curr, S_curr):
        X = np.hstack([L_curr, S_curr])
        next_L = [m.predict(X).reshape(-1, 1) for m in self.models]
        return np.column_stack(next_L)

# --- [Model 1] Binned HMM ---
class BinnedHMM(nn.Module):
    def __init__(self, n_covariates, bin_edges):
        super().__init__()
        self.register_buffer('bin_edges', bin_edges)
        self.n_bins = len(bin_edges) + 1
        self.psi = nn.Parameter(torch.tensor([0.8]))       
        self.gamma_cum = nn.Parameter(torch.tensor([0.02])) 
        self.gamma_G = nn.Parameter(torch.tensor([0.1]))    
        self.gamma_L = nn.Linear(n_covariates, 1, bias=False)
        self.beta_bins = nn.Parameter(torch.zeros(self.n_bins))
        self.beta_Z = nn.Parameter(torch.tensor([0.5]))
        self.beta_0 = nn.Parameter(torch.tensor([-4.0]))
        self.beta_L = nn.Linear(n_covariates, 1, bias=False)
        self.cov_model = CovariateTransitionModel(n_covariates)

    def _get_bin(self, S_cont):
        S_cont = S_cont.contiguous().squeeze(-1)
        idx = torch.bucketize(S_cont, self.bin_edges)
        return F.one_hot(idx.clamp(0, self.n_bins-1), self.n_bins).float()

    def fit(self, G, S, C, Y, L, n_epochs=50, batch_size=1024):
        self.cov_model.fit(L, S)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        dataset = TensorDataset(G, S, C, Y, L)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in tqdm(range(n_epochs)):
            for b_G, b_S, b_C, b_Y, b_L in loader:
                optimizer.zero_grad()
                Z = torch.zeros(b_S.shape[0], 1).to(b_S.device)
                b_bins = self._get_bin(b_S)
                loss = 0
                for t in range(b_S.shape[1]):
                    logit = self.beta_0 + self.beta_Z * Z + (b_bins[:,t,:] * self.beta_bins).sum(-1, keepdim=True) + self.beta_L(b_L[:, t, :])
                    prob = torch.sigmoid(logit.clamp(-10, 10))
                    mask = (b_Y[:, t, :] >= 0).float()
                    loss += ((-b_Y[:, t, :] * torch.log(prob+1e-8) - (1-b_Y[:, t, :]) * torch.log(1-prob+1e-8)) * mask).mean()
                    Z = self.psi * Z + self.gamma_cum * b_C[:, t, :] + self.gamma_G * b_G + self.gamma_L(b_L[:, t, :])
                loss.backward(); optimizer.step()
        return self

    def simulate(self, G, L_init, t_s_scaled, n_time=30, static_mode=False):
        self.eval(); n = G.shape[0]; device = G.device
        Z, surv = torch.zeros(n, 1).to(device), torch.ones(n, 1).to(device)
        curr_C = torch.zeros(n, 1).to(device)
        curr_L_np = L_init.cpu().numpy()
        
        target_bin = self._get_bin(torch.full((n, 1), t_s_scaled).to(device))
        S_val_np = np.full((n, 1), t_s_scaled)
        
        risk_traj = []
        with torch.no_grad():
            for t in range(n_time):
                curr_L_tensor = torch.tensor(curr_L_np, dtype=torch.float32).to(device)
                logit = self.beta_0 + self.beta_Z * Z + (target_bin * self.beta_bins).sum(-1, keepdim=True) + self.beta_L(curr_L_tensor)
                surv *= (1 - torch.sigmoid(logit))
                risk_traj.append(1 - surv.mean().item())
                curr_C += t_s_scaled
                Z = self.psi * Z + self.gamma_cum * curr_C + self.gamma_G * G + self.gamma_L(curr_L_tensor)
                if not static_mode:
                    curr_L_np = self.cov_model.predict(curr_L_np, S_val_np)
        return risk_traj
    
    def forward_filter(self, G, S, C, Y, L):
        self.eval(); Z_seq = []; Z = torch.zeros_like(S[:, 0, :])
        with torch.no_grad():
            for t in range(S.shape[1]):
                Z_seq.append(Z)
                Z = self.psi * Z + self.gamma_cum * C[:, t, :] + self.gamma_G * G + self.gamma_L(L[:, t, :])
        return torch.stack(Z_seq, dim=1)

# --- [Model 2] Continuous HMM (With Smoothing) ---
class ContinuousHMM(nn.Module):
    def __init__(self, n_covariates, n_bins=15):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer('bin_edges', torch.linspace(-2.5, 2.5, n_bins + 1))
        self.psi = nn.Parameter(torch.tensor([0.8]))       
        self.gamma_cum = nn.Parameter(torch.tensor([0.02])) 
        self.gamma_G = nn.Parameter(torch.tensor([0.1]))    
        self.gamma_L = nn.Linear(n_covariates, 1, bias=False)
        self.beta_bins = nn.Parameter(torch.zeros(n_bins))
        self.beta_Z = nn.Parameter(torch.tensor([0.5]))
        self.beta_0 = nn.Parameter(torch.tensor([-4.0]))
        self.beta_L = nn.Linear(n_covariates, 1, bias=False)
        self.cov_model = CovariateTransitionModel(n_covariates)

    def _get_bin(self, S_cont):
        S_cont = S_cont.contiguous().squeeze(-1)
        indices = torch.bucketize(S_cont, self.bin_edges[1:-1])
        return F.one_hot(indices.clamp(0, self.n_bins-1).long(), self.n_bins).float()

    def fit(self, G, S, C, Y, L, n_epochs=50, batch_size=1024, lambda_smooth=0.02):
        """lambda_smooth: Penalty for difference between adjacent bins"""
        self.cov_model.fit(L, S)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        dataset = TensorDataset(G, S, C, Y, L)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in tqdm(range(n_epochs)):
            for b_G, b_S, b_C, b_Y, b_L in loader:
                optimizer.zero_grad()
                Z = torch.zeros(b_S.shape[0], 1).to(b_S.device)
                b_bins = self._get_bin(b_S)
                loss = 0
                for t in range(b_S.shape[1]):
                    logit = self.beta_0 + self.beta_Z * Z + (b_bins[:,t,:] * self.beta_bins).sum(-1, keepdim=True) + self.beta_L(b_L[:, t, :])
                    prob = torch.sigmoid(logit.clamp(-10, 10))
                    mask = (b_Y[:, t, :] >= 0).float()
                    nll = ((-b_Y[:, t, :] * torch.log(prob+1e-8) - (1-b_Y[:, t, :]) * torch.log(1-prob+1e-8)) * mask).mean()
                    loss += nll
                    Z = self.psi * Z + self.gamma_cum * b_C[:, t, :] + self.gamma_G * b_G + self.gamma_L(b_L[:, t, :])
                
                diff = self.beta_bins[1:] - self.beta_bins[:-1]
                smooth_loss = lambda_smooth * torch.sum(diff**2)
                
                total_loss = loss + smooth_loss
                total_loss.backward()
                optimizer.step()
        return self

    def simulate(self, G, L_init, t_s_scaled, n_time=30, static_mode=False):
        self.eval(); n = G.shape[0]; device = G.device
        Z, surv = torch.zeros(n, 1).to(device), torch.ones(n, 1).to(device)
        curr_C = torch.zeros(n, 1).to(device)
        curr_L_np = L_init.cpu().numpy()
        target_bin = self._get_bin(torch.full((n, 1), t_s_scaled).to(device))
        S_val_np = np.full((n, 1), t_s_scaled)
        risk_traj = []
        with torch.no_grad():
            for t in range(n_time):
                curr_L_tensor = torch.tensor(curr_L_np, dtype=torch.float32).to(device)
                logit = self.beta_0 + self.beta_Z * Z + (target_bin * self.beta_bins).sum(-1, keepdim=True) + self.beta_L(curr_L_tensor)
                surv *= (1 - torch.sigmoid(logit))
                risk_traj.append(1 - surv.mean().item())
                curr_C += t_s_scaled
                Z = self.psi * Z + self.gamma_cum * curr_C + self.gamma_G * G + self.gamma_L(curr_L_tensor)
                if not static_mode:
                    curr_L_np = self.cov_model.predict(curr_L_np, S_val_np)
        return risk_traj
    
    def forward_filter(self, G, S, C, Y, L):
        self.eval(); Z_seq = []; Z = torch.zeros_like(S[:, 0, :])
        with torch.no_grad():
            for t in range(S.shape[1]):
                Z_seq.append(Z)
                Z = self.psi * Z + self.gamma_cum * C[:, t, :] + self.gamma_G * G + self.gamma_L(L[:, t, :])
        return torch.stack(Z_seq, dim=1)