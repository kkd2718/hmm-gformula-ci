```markdown
# HMM-based Parametric g-formula for CVD Risk Simulation

**Optimal Timing of Smoking Cessation Stratified by Polygenic Susceptibility: A Causal Inference Simulation Study**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

This repository contains the official implementation of the simulation framework used in the Master's Thesis: **"Optimal Timing of Smoking Cessation Based on Polygenic Susceptibility to Cardiovascular Disease."**

We propose a novel causal inference framework integrating **Hidden Markov Models (HMM)** with the **Parametric g-formula** to handle time-varying confounding and unobserved latent health states (e.g., "sick-quitter" effect). The simulation is rigorously calibrated to real-world epidemiological statistics of the Korean population, utilizing data from **KoGES**, **KNHANES 2023**, and **KDCA 2022**.

### Key Features
* **Hidden Markov Modeling (HMM):** Captures unobserved latent health states ($Z_t$) to correct bias in smoking cessation effects.
* **Parametric g-formula:** Estimates counterfactual cumulative risks under dynamic interventions (e.g., "Quit at age 50").
* **Real-World Calibration:** Simulation parameters are fine-tuned to reflect Korean gender gaps in smoking rates (Male ~37%, Female ~8%) and CVD incidence (10-year risk 5~10%), including **aging effects**.
* **Gene-Environment Interaction (GxE):** Stratified analysis by Polygenic Risk Score (PRS) to identify optimal intervention timing.

---

## 📂 Repository Structure

```text
hmm-gformula-cvd/
├── config.py                 # Simulation parameters (calibrated to KNHANES/KoGES)
├── data_generator.py         # Synthetic data generation with aging effects
├── main.py                   # Main entry point for basic validation experiments
├── analysis_advanced_prs.py  # Advanced analysis (Spline curves with 95% CI)
├── models/
│   └── hmm_gformula.py       # Core HMM and g-formula logic (PyTorch)
├── experiments/
│   └── run_experiments.py    # Experiment pipelines (Bootstrap CI, Robustness)
├── results/                  # Output figures and tables
├── requirements.txt          # Python dependencies
└── README.md

```

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.8+
* PyTorch
* NumPy, Pandas
* Matplotlib, Seaborn, SciPy
* Tqdm

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/hmm-gformula-cvd.git
cd hmm-gformula-cvd
pip install -r requirements.txt

```

---

## 💻 Usage

### 1. Methodological Validation (Table 1-3)

Run the full simulation suite to validate parameter recovery, check sample size robustness, and compare with conventional methods (e.g., Logistic Regression).

```bash
python main.py --full

```

### 2. Causal Inference & Clinical Analysis (Figure 3-4)

Run the advanced analysis to generate **Spline Curves with 95% Confidence Intervals (Shaded Area)**. This script analyzes:

* **Effect Modification:** How Risk Ratio changes across PRS levels.
* **Urgency of Cessation:** How Risk Ratio increases as cessation is delayed.

```bash
python analysis_advanced_prs.py

```

*Output:* The results (Figures and CSVs) will be saved in the `./results/` directory.

---

## 📊 Visualization Outputs

| **Figure A. PRS Effect Modification** | **Figure B. Urgency of Cessation** |
| --- | --- |
| Demonstrates the varying risk ratios across the continuous PRS spectrum. | Shows the diminishing returns of delayed smoking cessation. |
| *(Generated via analysis_advanced_prs.py)* | *(Generated via analysis_advanced_prs.py)* |

---

## 🛠 Configuration

You can modify the simulation parameters in `config.py` to adapt to different populations (e.g., UK Biobank settings).

* **`TRUE_PARAMS_EXPOSURE`**: Adjust smoking prevalence, persistence (`alpha_S`), and aging effects (`alpha_time`).
* **`TRUE_PARAMS_OUTCOME`**: Adjust baseline CVD risk (`beta_0`) and aging risk (`beta_time`).
* **`TRUE_PARAMS_HIDDEN_STATE`**: Modify the transition dynamics of latent health.

Current settings are calibrated to **2023 Korean Health Statistics**:

* Male Smoking Rate: ~37% (decreasing with age)
* CVD Incidence: 10-year cumulative risk ~5-10%
