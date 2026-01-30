# HMM-based Parametric g-formula for CVD Risk Simulation

**Estimating the Causal Effect of Smoking Cessation Stratified by Polygenic Risk: A Simulation Study using Hidden Markov Models.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## 📌 Overview

This repository contains the official implementation of the simulation framework used in the Master's Thesis: **"Estimating the Causal Effect of Smoking Cessation Stratified by Polygenic Risk: A Simulation Study using Hidden Markov Models."**

We propose a novel causal inference framework integrating **Hidden Markov Models (HMM)** with the **Parametric g-formula** to handle time-varying confounding and unobserved latent health states (e.g., "sick-quitter" effect). The simulation is rigorously calibrated to **long-term epidemiological trends** of the Korean population (2001–2020), utilizing baseline data from **KoGES** and historical trends from **KNHANES**. Unlike static models, this framework reconstructs a **20-year longitudinal history**, capturing the dynamic decline in smoking rates and the cumulative incidence of CVD over two decades.

### Key Features
* **Hidden Markov Modeling (HMM):** Captures unobserved latent health states ($Z_t$) to correct bias in smoking cessation effects.
* **Parametric g-formula:** Estimates counterfactual cumulative risks under dynamic interventions (e.g., "Quit at age 50").
* **Real-World Calibration:** Simulation parameters are fine-tuned to reflect Korean gender gaps in smoking rates (Male -37%, Female -8%) and CVD incidence (10-year risk 5-10%), including **aging effects**.
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
git clone https://github.com/kkd2718/hmm-gformula-ci.git
cd hmm-gformula-ci
pip install -r requirements.txt

```

---

## 💻 Usage

### 1. Methodological Validation (Table 1-3)

Run the full simulation suite to validate parameter recovery, check sample size robustness, and compare with conventional methods (e.g., Logistic Regression).

```bash
python main.py --full

```

### 2. Causal Inference & Clinical Analysis (Figure 1-2)

Run the advanced analysis to generate **Spline Curves with 95% Confidence Intervals (Shaded Area)**. This script analyzes:

* **Effect Modification:** How Risk Ratio changes across PRS levels.
* **Urgency of Cessation:** How Risk Ratio increases as cessation is delayed.

```bash
python main.py --advanced

```

*Output:* The results (Figures and CSVs) will be saved in the `./results/` directory.

---

## 📊 Visualization Outputs

The simulation results demonstrate strong **Gene-Environment Interaction** and the critical importance of **early intervention**.

### **Figure A. Synergistic Effect of PRS and Smoking**
> **"High genetic risk amplifies the harm of smoking."**

![Figure A: PRS Effect Modification](./results/curve_a_prs_effect_95ci.png)

* **Interpretation:** The Risk Ratio (RR) of smoking increases sharply as the Polygenic Risk Score (PRS) increases. This "fanning out" pattern indicates a **synergistic interaction**, meaning high-risk individuals suffer disproportionately more from smoking compared to low-risk individuals.

<br>

### **Figure B. Urgency of Cessation (Timing Effect)**
> **"Every year of delay matters."**

![Figure B: Quit Timing Effect](./results/curve_b_quit_timing_95ci.png)

* **Interpretation:** Delaying smoking cessation shows a **linear increase** in CVD risk ratio. There is no "safe window" for delay; the simulation confirms that **quitting immediately** yields the greatest preventive benefit compared to postponing even by a single year.

---

## 🛠 Configuration

You can modify the simulation parameters in `config.py` to adapt to different populations (e.g., UK Biobank settings).

* **`TRUE_PARAMS_EXPOSURE`**: Controls smoking dynamics.
    * `alpha_S`: Smoking persistence (addiction).
    * `alpha_time`: **Time-dependent trend** (reflecting the historical decrease in smoking rates and aging effects).
* **`TRUE_PARAMS_OUTCOME`**: Controls CVD risk.
    * `beta_0`: Baseline risk.
    * `beta_time`: **Aging effect** (yearly increase in CVD risk due to aging).
* **`TRUE_PARAMS_HIDDEN_STATE`**: Transitions of the latent health state ($Z_t$).

Current settings are calibrated to **KoGES (2001-2020) & KNHANES trends**:

* **Simulation Period:** 20 years (Long-term follow-up).
* **Male Smoking Rate:** approx. **49%** (baseline) $\rightarrow$ **42%** (decreasing trend via `alpha_time`).
* **CVD Incidence:** 20-year cumulative risk **5-10%** (calibrated via `beta_0` & `beta_time`).