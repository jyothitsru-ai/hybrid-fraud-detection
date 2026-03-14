# hybrid-fraud-detection
Hybrid Deep Learning + XGBoost + SHAP for Real-Time Financial Fraud Detection
# 🔍 Hybrid Deep Learning & Explainable AI Framework for Real-Time Financial Fraud Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-SRU-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



## 📄 Abstract

This repository contains the official implementation of a hybrid fraud detection framework that combines **deep neural network embeddings**, **gradient-boosted ensemble classifiers**, and **SHAP-based explainability** for real-time financial transaction fraud detection.

The framework addresses three key challenges in production fraud detection:
- **Class imbalance** — fewer than 0.2% of real-world transactions are fraudulent
- **Model opacity** — regulatory compliance demands transparent, auditable decisions
- **Latency constraints** — authorization systems require single-digit millisecond inference

Evaluated on three complementary datasets, the system achieves ROC-AUC of **0.994** on the European credit card benchmark, while maintaining sub-millisecond inference on commodity CPUs and providing per-transaction SHAP explanations.

---

## 🏗️ Architecture

```
Raw Transaction Stream
        │
        ▼
┌───────────────────────────────────────────────────┐
│               FEATURE BUILDER                     │
│  Tabular snapshot │ Temporal window │ Entity graph │
└──────────┬────────────────┬─────────────────┬─────┘
           │                │                 │
           ▼                ▼                 ▼
    ┌──────────┐    ┌──────────────┐   ┌────────────┐
    │  Tabular  │    │   Temporal   │   │   Graph    │
    │  Encoder  │    │   Encoder    │   │  Encoder   │
    │   (MLP)   │    │(Transformer/ │   │(GraphSAGE) │
    │           │    │    TCN)      │   │            │
    └─────┬─────┘    └──────┬───────┘   └─────┬──────┘
          │                 │                  │
          └────────┬─────────────────┬─────────┘
                   ▼                 ▼
          ┌────────────────────────────────┐
          │         FUSION BLOCK           │
          │  (Layer Norm + Gating Unit)    │
          │  → Latent representation z     │
          └────────────────┬───────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │    GRADIENT-BOOSTED ENSEMBLE   │
          │    (XGBoost / LightGBM)        │
          │    Cost-sensitive classification│
          │    → Fraud score  s ∈ [0,1]    │
          └────────────────┬───────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
   ┌───────────────────┐   ┌───────────────────────┐
   │  ISOTONIC CALIB.  │   │   SHAP EXPLAINER      │
   │  Post-hoc         │   │   (TreeSHAP)          │
   │  calibration      │   │   Per-transaction      │
   │  ECE < 2%         │   │   feature attribution  │
   └───────────────────┘   └───────────────────────┘
               │
               ▼
    ┌──────────────────┐
    │  FRAUD DECISION  │
    │  + Reason codes  │
    │  (< 5ms latency) │
    └──────────────────┘
```

**Pipeline summary:**
1. Raw transaction features → three parallel encoders (tabular MLP, temporal Transformer/TCN, graph GraphSAGE)
2. Fused latent vector `z` → XGBoost/LightGBM ensemble classifier
3. Isotonic regression calibrates output probabilities
4. TreeSHAP generates per-transaction explanations
5. Cost-optimal threshold applied for final binary decision

---

## 📊 Datasets

This project uses **three publicly available datasets**. Raw data files are **not included** in this repository. Please download them from the links below.

### 1. European Credit Card Fraud Dataset (Kaggle)
- **Records:** 284,807 transactions over 2 days
- **Fraud rate:** 0.172% (492 fraudulent)
- **Features:** 28 PCA-anonymized features + Time + Amount
- **Download:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **License:** [DbCL v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

### 2. IEEE-CIS Fraud Detection Dataset (Kaggle)
- **Records:** ~590,000 transactions
- **Features:** 400+ (transaction metadata, device fingerprints, identity signals)
- **Challenges:** High cardinality categoricals, extreme sparsity, temporal leakage risk
- **Download:** https://www.kaggle.com/c/ieee-fraud-detection/data
- **Note:** Requires Kaggle account and competition rules acceptance

### 3. PaySim Mobile Money Simulator
- **Records:** ~6.3 million synthetic transactions
- **Fraud types:** Identity theft, account collusion, abnormal transaction flows
- **Key property:** Built-in temporal drift — fraud patterns shift across simulation steps
- **Download:** https://www.kaggle.com/datasets/ealaxi/paysim1

After downloading, place files in the `data/` directory:
```
data/
├── creditcard.csv          # from Kaggle (Credit Card dataset)
├── train_transaction.csv   # from Kaggle (IEEE-CIS)
├── train_identity.csv      # from Kaggle (IEEE-CIS)
└── PS_20174392719_1491204439457_log.csv  # PaySim
```

---

## ⚙️ Installation

**Requirements:** Python 3.9+ recommended.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/hybrid-fraud-detection.git
cd hybrid-fraud-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies overview

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.0.0 | Deep neural network encoder |
| `xgboost` | ≥2.0.0 | Gradient-boosted ensemble |
| `lightgbm` | ≥4.0.0 | Alternative ensemble head |
| `shap` | ≥0.44.0 | TreeSHAP explainability |
| `scikit-learn` | ≥1.3.0 | Preprocessing, calibration, metrics |
| `imbalanced-learn` | ≥0.11.0 | SMOTE oversampling |
| `optuna` | ≥3.4.0 | Bayesian hyperparameter optimization |
| `fastapi` | ≥0.104.0 | REST API for real-time inference |
| `pandas` | ≥2.0.0 | Data manipulation |

---

## 🚀 Usage

### Quick start — reproduce main results

```bash
# Step 1: Preprocess all three datasets
python src/preprocessing.py --dataset creditcard --input data/creditcard.csv

# Step 2: Train the hybrid model
python src/model.py --dataset creditcard --epochs 50 --embedding-dim 64

# Step 3: Train ensemble on DNN embeddings
python src/ensemble.py --dataset creditcard --model xgboost --optimize

# Step 4: Calibrate and evaluate
python src/calibration.py --dataset creditcard

# Step 5: Generate SHAP explanations
python src/shap_explainer.py --dataset creditcard --output results/figures/
```

### Run all experiments (all three datasets)

```bash
python run_experiments.py --all-datasets --save-results
```

Results will be saved to `results/` and figures to `results/figures/`.

### Real-time inference API

```bash
# Start FastAPI inference server
uvicorn src.inference:app --host 0.0.0.0 --port 8000

# Send a test transaction (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 249.62, "V1": -1.36, "V2": -0.07, ...}'

# Response:
# {
#   "fraud_score": 0.0023,
#   "decision": "legitimate",
#   "shap_explanation": {"V14": -0.41, "V10": -0.18, ...},
#   "latency_ms": 3.2
# }
```

### Jupyter notebooks (step-by-step)

```bash
jupyter notebook notebooks/
```

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis, class imbalance visualization |
| `02_preprocessing.ipynb` | SMOTE, feature engineering, normalization |
| `03_model_training.ipynb` | DNN training, embedding extraction, ensemble fitting |
| `04_evaluation_shap.ipynb` | Full evaluation + SHAP global/local explanations |

---

## 📈 Results

### Table 1 — Comparative performance across models

| Model | Dataset | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Credit Card | 97.3% | 78.2% | 69.5% | 73.6% | 0.956 | 0.74 |
| Random Forest | Credit Card | 98.9% | 87.1% | 88.4% | 87.7% | 0.981 | 0.82 |
| XGBoost | Credit Card | 99.1% | 89.2% | 91.3% | 90.2% | 0.988 | 0.86 |
| Standalone DNN | Credit Card | 98.7% | 85.1% | 86.9% | 86.0% | 0.975 | 0.81 |
| **Hybrid DNN+XGBoost** | **Credit Card** | **99.3%** | **91.3%** | **93.7%** | **92.5%** | **0.994** | **0.91** |
| **Hybrid DNN+XGBoost** | **IEEE-CIS** | **98.2%** | **88.7%** | **90.2%** | **89.4%** | **0.980** | **0.84** |
| **Hybrid DNN+XGBoost** | **PaySim** | **98.9%** | **90.0%** | **92.1%** | **91.0%** | **0.987** | **0.88** |

### Table 2 — Effect of calibration on probabilistic stability

| Dataset | Metric | Pre-calibration | Post-calibration |
|---|---|---|---|
| Credit Card | Expected Calibration Error (ECE) | 6.1% | 2.0% |
| Credit Card | Brier Score | 0.048 | 0.032 |
| IEEE-CIS | Expected Calibration Error (ECE) | 7.5% | 2.5% |
| PaySim | Expected Calibration Error (ECE) | 5.8% | 1.9% |

### Table 3 — Ablation study (Credit Card dataset)

| Configuration | ROC-AUC | PR-AUC | Recall | F1-score |
|---|---|---|---|---|
| Ensemble only (no DNN embeddings) | 0.985 | 0.85 | 89.4% | 88.6% |
| DNN + Linear head (no ensemble) | 0.978 | 0.82 | 86.5% | 85.9% |
| Full hybrid (no focal reweighting) | 0.990 | 0.87 | 90.1% | 89.7% |
| **Full hybrid (proposed)** | **0.994** | **0.91** | **93.7%** | **92.5%** |

---

## 📁 Repository Structure

```
hybrid-fraud-detection/
├── data/
│   └── README.md               # Dataset download instructions
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_shap.ipynb
├── src/
│   ├── preprocessing.py        # SMOTE, feature engineering, normalization
│   ├── model.py                # DNN encoder (tabular + temporal + graph)
│   ├── ensemble.py             # XGBoost/LightGBM classifier head
│   ├── shap_explainer.py       # TreeSHAP global + local explanations
│   ├── calibration.py          # Isotonic regression + ECE evaluation
│   └── inference.py            # FastAPI real-time serving
├── results/
│   └── figures/                # ROC curves, PR curves, SHAP summary plots
├── requirements.txt
├── run_experiments.py          # End-to-end experiment runner
├── README.md
└── LICENSE
```

---

## 🔬 Reproducibility Notes

- All results reported as **mean ± std over 5 time-stratified folds**
- Statistical significance tested with **paired bootstrap resampling** vs. baselines
- Random seeds fixed: `numpy.random.seed(42)`, `torch.manual_seed(42)`
- Hyperparameter search space and optimal values logged in `results/hparams.json` after running experiments
- Latency profiling performed on **Intel Core i7-11800H CPU**, no GPU required for inference

---

## 📋 Data Availability Statement

The source code for this study is publicly available at:
**https://github.com/YOUR_USERNAME/hybrid-fraud-detection**

The datasets used in this study are all publicly available:
- European Credit Card Fraud dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- IEEE-CIS Fraud Detection dataset: https://www.kaggle.com/c/ieee-fraud-detection
- PaySim Mobile Money Simulator: https://www.kaggle.com/datasets/ealaxi/paysim1

No proprietary or restricted data were used in this study.

---

## 📝 Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{jyothi2025hybridfraud,
  title     = {A Hybrid Deep Learning and Explainable AI Framework for
               Real-Time Financial Fraud Detection in Dynamic Environments},
  author    = {Jyothi, T. and {others}},
  journal   = {JOURNAL NAME},
  year      = {2025},
  volume    = {},
  pages     = {},
  doi       = {10.XXXX/XXXXXXX},
  url       = {https://github.com/jyothitsru-ai/hybrid-fraud-detection}
}
```

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **SRU License** — see the [LICENSE](LICENSE) file for details.

The datasets used in experiments are governed by their respective licenses (Kaggle Terms of Service, DbCL v1.0). Please comply with those terms when downloading and using the data.

---

## Acknowledgements

This work builds on the following open-source projects: [XGBoost](https://github.com/dmlc/xgboost), [LightGBM](https://github.com/microsoft/LightGBM), [SHAP](https://github.com/slundberg/shap), [PyTorch](https://pytorch.org), [imbalanced-learn](https://imbalanced-learn.org), and [FastAPI](https://fastapi.tiangolo.com).
