# 🫀 ECG Arrhythmia Classification — CNN+BiLSTM+Attention vs 2D-CNN Baseline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![MIT-BIH](https://img.shields.io/badge/Dataset-MIT--BIH%20Arrhythmia-red)](https://physionet.org/content/mitdb/1.0.0/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Paper%20Ready-brightgreen)]()

> **Automatic ECG arrhythmia classification using STFT spectrograms with a novel CNN+BiLSTM+Attention architecture — benchmarked against a 2D-CNN baseline on the full MIT-BIH database (all 48 records, 5 classes, real imbalanced test set).**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results at a Glance](#-results-at-a-glance)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Pipeline](#-pipeline)
- [Key Figures](#-key-figures)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

This repository contains the full training and evaluation pipeline for a deep learning system that classifies ECG beats into 5 arrhythmia categories. The core contribution is a **CNN+BiLSTM+Attention** model that combines spatial feature extraction (via 2D convolution on STFT spectrograms) with temporal modelling (via BiLSTM) and a learnable attention mechanism — compared against a lightweight **2D-CNN baseline**.

**Key design choices:**
- Spectrograms are computed via Short-Time Fourier Transform (STFT) on segmented ECG beats
- Training uses **augmented balanced data** (50,000 beats); validation and test use **real imbalanced data only** — zero leakage guaranteed
- EarlyStopping monitors the real validation set, keeping the test set completely held out until final evaluation
- Explicit class weighting to handle the severe natural class imbalance (NOR: 75,011 vs APC: 2,546)

---

## 📊 Results at a Glance

Evaluated on the **real imbalanced test set** (20% hold-out, MIT-BIH all 48 records):

| Metric | Model 1: 2D-CNN | Model 2: CNN+BiLSTM+Attn |
|---|---|---|
| **Overall Accuracy** | 98.47% | **98.95%** |
| **Macro F1-Score** | 95.70% | **96.99%** |
| **AUC-ROC (macro)** | 0.9984 | **0.9989** |
| NOR F1 | 99.1% | **99.4%** |
| LBB F1 | 98.8% | **99.2%** |
| RBB F1 | 99.1% | **99.5%** |
| PVC F1 | 96.1% | **97.3%** |
| **APC F1 ⭐** | 85.5% | **89.6%** |

> ⭐ **APC (Atrial Premature Contraction)** is the minority class (only 2.55% of real beats) and the most clinically critical — our proposed model closes the gap significantly.

---

## 🗃️ Dataset

**Source:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — all 48 records

**5 Classes:**

| Label | Class | Raw Count | % of Total |
|---|---|---|---|
| NOR | Normal Sinus Rhythm | 75,011 | 74.9% |
| LBB | Left Bundle Branch Block | 8,071 | 8.1% |
| RBB | Right Bundle Branch Block | 7,255 | 7.2% |
| PVC | Premature Ventricular Contraction | 7,129 | 7.1% |
| APC | Atrial Premature Contraction | 2,546 | 2.5% |

**Data Split (70 / 10 / 20):**

| Split | Beats | Type | Purpose |
|---|---|---|---|
| Train | 50,000 | Augmented + balanced (10k/class) | Model learning |
| Validation | ~10,053 | Real imbalanced | EarlyStopping / ReduceLR |
| Test | ~20,001 | Real imbalanced | Final honest evaluation only |

---

## 🧠 Architecture

### Model 1 — 2D-CNN Baseline

```
Input (64×64×1)
  → Conv2D(8, 4×4) → BN → MaxPool
  → Conv2D(13, 2×2) → BN → MaxPool
  → Conv2D(13, 2×2) → BN → MaxPool
  → Flatten → Dense(128) → Dropout(0.5)
  → Softmax(5)
```

### Model 2 — CNN+BiLSTM+Attention (Proposed)

```
Input (64×64×1)
  → Conv2D(32, 3×3) → BN → MaxPool
  → Conv2D(64, 3×3) → BN → MaxPool
  → Conv2D(128, 3×3) → BN → MaxPool
  → Reshape → BiLSTM(128, return_sequences=True)
  → Attention (tanh score → softmax weights → weighted sum)
  → Dense(128) → Dropout(0.4)
  → Dense(64) → Dropout(0.3)
  → Softmax(5)
```

The attention mechanism learns to focus on the most discriminative temporal positions in the BiLSTM output — particularly beneficial for short-duration, morphologically subtle beats like APC.

---

## ⚙️ Pipeline

```
Raw ECG Signal (MIT-BIH)
        │
        ▼
   Beat Segmentation
        │
        ▼
  STFT Spectrogram (64×64)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  [Train Split]                 [Val + Test Splits]
  Augmentation                  Real beats only
  + Balancing                   (no augmentation)
  (10k per class)
        │
        ▼
  Model Training
  (class weights + EarlyStopping on Val)
        │
        ▼
  Final Evaluation on Test Set only
```

---

## 📈 Key Figures

<table>
<tr>
<td align="center"><b>Fig. 1 — Dataset Split</b><br><img src="paper_figures_v2/Fig1_Class_Distribution.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>Fig. 2 — Training Curves</b><br><img src="paper_figures_v2/Fig2_Training_Curves.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>Fig. 3 — Confusion Matrices</b><br><img src="paper_figures_v2/Fig3_Confusion_Matrices.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>Fig. 4 — Per-Class F1 Comparison</b><br><img src="paper_figures_v2/Fig4_F1_Comparison.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>Fig. 5 — ROC Curves</b><br><img src="paper_figures_v2/Fig5_ROC_Curves.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>Fig. 6 — Overall Metrics</b><br><img src="paper_figures_v2/Fig6_Overall_Metrics.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>Fig. 7 — Precision & Recall</b><br><img src="paper_figures_v2/Fig7_Precision_Recall.png" width="100%"></td>
</tr>
</table>

---

## 🛠️ Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/your-username/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification

pip install tensorflow numpy scikit-learn matplotlib seaborn pillow
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow>=2.10
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
pillow>=9.0
```

---

## 🚀 Usage

### 1. Prepare your data

Place the following `.npy` files in your data directory (set `data_dir` in CONFIG):

```
balanced_stft_final/
├── X_train.npy   # (50000, 64, 64) — augmented balanced spectrograms
├── y_train.npy   # (50000,)        — integer class labels
├── X_val.npy     # (~10053, 64, 64) — real val beats
├── y_val.npy     # (~10053,)
├── X_test.npy    # (~20001, 64, 64) — real test beats
└── y_test.npy    # (~20001,)
```

### 2. Configure paths

Edit the `CONFIG` block in `train_and_figures_v2.py`:

```python
CONFIG = {
    'data_dir'   : 'path/to/balanced_stft_final',
    'output_dir' : 'path/to/results_v2',
    'figures_dir': 'path/to/paper_figures_v2',
    'img_size'   : 64,
    'n_classes'  : 5,
    'batch_size' : 32,
    'epochs'     : 50,
    'lr'         : 0.001,
    'random_seed': 42,
    'dpi'        : 300,
}
```

### 3. Train and evaluate

```bash
python train_and_figures_v2.py
```

This single script will:
- Load all 6 data arrays
- Build both models
- Train with EarlyStopping (monitoring real val set)
- Evaluate on held-out test set
- Generate all 10 paper figures at 300 DPI

---

## 📁 Project Structure

```
ecg-arrhythmia-classification/
│
├── train_and_figures_v2.py      # Main training + figure generation script
│
├── balanced_stft_final/         # Input data (not tracked — see Usage)
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   └── y_test.npy
│
├── results_v2/                  # Saved model checkpoints
│   ├── Model1_2DCNN_best.keras
│   └── Model2_BiLSTM_Attn_best.keras
│
├── paper_figures_v2/            # All 10 publication-ready figures (300 DPI)
│   ├── Fig1_Class_Distribution.png
│   ├── Fig2_Training_Curves.png
│   ├── Fig3_Confusion_Matrices.png
│   ├── Fig4_F1_Comparison.png
│   ├── Fig5_ROC_Curves.png
│   ├── Fig6_Overall_Metrics.png
│   ├── Fig7_Precision_Recall.png
│   ├── Fig8_Paper_Comparison.png
│   ├── Fig9_APC_Analysis.png
│   └── Fig10_Summary_Dashboard.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Methodology

### Preprocessing
- ECG beats are segmented around R-peaks using the MIT-BIH annotations
- Each beat is converted to a **64×64 STFT spectrogram** (grayscale, normalised to [0, 1])

### Augmentation (Training Set Only)
- Random time-shift, amplitude scaling, and Gaussian noise injection applied to minority classes
- Final training set: 10,000 beats per class (50,000 total), perfectly balanced
- Validation and test sets are **never augmented**

### Training Strategy
- Optimizer: Adam (lr=0.001)
- Loss: Categorical cross-entropy with class weights (computed on training labels)
- EarlyStopping: monitors `val_accuracy`, patience=10, restores best weights
- ReduceLROnPlateau: monitors `val_loss`, factor=0.5, patience=5, min_lr=1e-6
- ModelCheckpoint: saves best model by `val_accuracy`

### Evaluation
- All reported metrics are computed on the **real imbalanced test set exclusively**
- No test data was ever used during training or hyperparameter selection

---

## 📚 Comparison with Literature

| Method | Accuracy | Notes |
|---|---|---|
| Huang 2019 (Original) | 99.00% | Train/test overlap likely |
| Ullah 2020 | 99.11% | Augmented test set |
| SE-CNN 2025 | 99.13% | Augmented test set |
| CNN-LSTM 2024 | 98.83% | Partial MIT-BIH records |
| **Our Model 1 (2D-CNN)** | **98.47%** | Real imbalanced test — stricter |
| **Our Model 2 (Proposed)** | **98.95%** | Real imbalanced test — stricter |

> ⚠️ Our evaluation protocol is **more rigorous** than most prior work: test set is fully held out, real imbalanced, and never touched during training or model selection.

---

## 📄 Citation

If you use this code or results in your research, please cite:

```bibtex
@article{yourname2025ecg,
  title   = {ECG Arrhythmia Classification Using CNN+BiLSTM+Attention on STFT Spectrograms},
  author  = {Your Name and Co-Authors},
  journal = {Journal Name},
  year    = {2025},
  note    = {GitHub: https://github.com/your-username/ecg-arrhythmia-classification}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — Moody & Mark, PhysioNet
- [TensorFlow / Keras](https://www.tensorflow.org/) — Model implementation
- [scikit-learn](https://scikit-learn.org/) — Evaluation metrics

---

<p align="center">
  Made with ❤️ for better cardiac diagnostics
</p>
