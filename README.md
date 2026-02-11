# Dynamic Causal Inference in Healthcare: From Simulation to Intensive Care
### Correcting Time-Varying Confounding via Latent State Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

This repository contains the official implementation of the Master's Thesis: **"Dynamic Causal Inference of Mechanical Power in Severe ARDS: A Latent State Modeling Approach."**

We propose a **Continuous-Time Hidden Markov Model (HMM)** framework to address **Time-Varying Confounding** (e.g., Weaning Bias, Sick-Quitter Bias) in longitudinal medical data. This repository consists of two main studies:

1.  **Part I (Methodological Validation):** Verifying the model using a calibrated **Smoking-CVD Simulation**.
2.  **Part II (Clinical Application):** Investigating the causal effect of Mechanical Power in ARDS patients using **MIMIC-IV**.

---

## 📐 Methodology: Implementation Details

To overcome the limitations of static models (which ignore temporal confounding), we implemented a **Dynamic Latent State Model** using PyTorch.

### 1. Latent State Transition (Bias Correction)
We infer a latent physiological state $Z_t$ (e.g., true lung severity) that evolves over time based on previous state, intervention, and covariates.

$$Z_t = \sigma(\mathbf{W}_z Z_{t-1} + \mathbf{W}_a A_{t-1} + \mathbf{W}_x \mathbf{X}_{t-1})$$

```python
# models/dynamic_hmm.py

class ContinuousHMM(nn.Module):
    def forward(self, ...):
        # ...
        # Update Latent State (Recurrent Step)
        # Z: Latent State, G: Genetics, L: Covariates, C: Cumulative Load
        Z = torch.sigmoid(
            self.psi * Z_prev +       # W_z * Z_{t-1}
            self.gamma_G * G +        # W_g * G
            self.gamma_L(L_curr) +    # W_x * X_{t-1}
            self.gamma_cum * C_curr   # Cumulative Exposure Effect
        )

```

### 2. Outcome Model with Smoothness Regularization

The mortality risk  is modeled as a function of the latent state and the intervention. We use **Smoothness Regularization** to detect non-linear causal effects (e.g., U-shape) without overfitting.

```python
# models/dynamic_hmm.py

    def compute_loss(self, ...):
        # 1. Negative Log Likelihood
        logit = self.beta_0 + self.beta_Z * Z + self.beta_bins(target_bin)
        loss_nll = self.criterion(logit, Y)
        
        # 2. Smoothness Regularization (Key for U-Shape Detection)
        # Penalize the difference between adjacent bin coefficients
        diff = self.beta_bins.weight[1:] - self.beta_bins.weight[:-1]
        loss_smooth = self.lambda_smooth * (diff ** 2).sum()
        
        return loss_nll + loss_smooth

```

---

## 🧬 Part I: Methodological Validation (Simulation)

> **"Can HMM recover true causal effects in the presence of unmeasured confounding?"**

Before the clinical application, we validated the model using a **Smoking Cessation & CVD** simulation calibrated to the Korean population (KoGES/KNHANES). The goal was to correct for "Sick-Quitter Bias" (reverse causality).

### Validation Results

* **Curve A (PRS Effect):** Demonstrates how genetic risk modifies the causal effect of smoking.
* **Curve B (Quit Timing):** Quantifies the diminishing benefit of cessation as it is delayed.

| PRS Effect Modification | Quit Timing Urgency |
| :---: | :---: |
| ![Curve A](study_1_koges_simulation/results/curve_a_prs_effect_95ci.png) | ![Curve B](study_1_koges_simulation/results/curve_b_quit_timing_95ci.png) |

---

## 🏥 Part II: Clinical Application (Severe ARDS)

> **"Does lower Mechanical Power always lead to better survival?"**

We applied our HMM framework to **Severe ARDS patients** in the MIMIC-IV database. Unlike traditional models (Cox, Standard G-formula) that assume a linear relationship ("Lower is Better"), our model identified a distinct **U-Shaped Causal Relationship**.

### Key Findings

* **Optimal Range:** 12–13 J/min (Recruitment Benefit).
* **HMM vs Baselines:** Standard models fail to capture the risk of low MP due to weaning bias.

| Main Result (HMM vs Baselines) | Robustness Check |
| :---: | :---: |
| ![Figure 2A](study_2_ards_mimic/results/Figure2A_Main_Result.png) | ![Figure 2B](study_2_ards_mimic/results/Figure2B_Robustness.png) |
| **Figure 2A.** The HMM (Red) reveals a U-shape curve, whereas Cox/G-formula (Black/Green) show linear trends. | **Figure 2B.** Comparison with categorical baselines confirms the stability of the HMM's smooth curve. |

---

## 📂 Repository Structure

```bash
hmm-gformula-ci/
├── models/                       # [Core] Shared Models
│   ├── dynamic_hmm.py            # Continuous & Binned HMM (PyTorch)
│   └── baseline_methods.py       # Cox, Logistic, G-formula
│
├── study_1_koges_simulation/     # [Part I] Method Validation
│   ├── main.py                   # Simulation Entry Point
│   ├── config.py                 # Parameters (KoGES Calibrated)
│   ├── modules/                  # Data Generation Logic
│   ├── experiments/              # Experiment Scenarios
│   ├── utils/                    # Metrics & Visualization
│   └── results/                  # Validation Figures
│
└── study_2_ards_mimic/           # [Part II] Clinical Application
    ├── main.py                   # Main Analysis (Fig 2, 3)
    ├── make_table1.py            # Generate Table 1
    ├── preprocessing/            # MIMIC Extraction & Tensor Conversion
    │   ├── mimic_extract.py
    │   └── hmm_processing.py
    └── results/                  # Thesis Figures

```

---

## 🚀 Usage

### 1. Run Simulation Validation (Part I)

```bash
# Run full validation suite (All experiments)
python study_1_koges_simulation/main.py --all

# Run specific advanced analysis (Spline Curves)
python study_1_koges_simulation/main.py --advanced

```

*Outputs will be saved in `study_1_koges_simulation/results/*`

### 2. Run ARDS Analysis (Part II)

*Note: Access to MIMIC-IV database is required.*

```bash
# 1. Preprocessing (Extraction & Tensor Conversion)
python study_2_ards_mimic/preprocessing/mimic_extract.py
python study_2_ards_mimic/preprocessing/hmm_processing.py

# 2. Generate Table 1 (Baseline Characteristics)
python study_2_ards_mimic/make_table1.py

# 3. Run Main Experiments (HMM Training & Simulation)
python study_2_ards_mimic/main.py

```

*Outputs will be saved in `study_2_ards_mimic/results/*`

---

## 📝 Citation

If you find this code useful, please cite:

> **Kiduk Kim et al.**. "Dynamic Causal Inference of Mechanical Power in Severe ARDS: A Latent State Modeling Approach." Master's Thesis, Graduate School of Public Health, Yonsei University, 2026.
