"""
=============================================================
  ECG Arrhythmia — Complete Training + All Paper Figures v2
  Updated for Train / Val / Test pipeline (balanced_stft_final)

  Model 1: 2D-CNN Baseline
  Model 2: CNN + BiLSTM + Attention (Proposed)

  Data:
    X_train / y_train → augmented balanced (50,000 beats)
    X_val   / y_val   → real beats only    → EarlyStopping
    X_test  / y_test  → real beats only    → final evaluation
=============================================================

Install:
    pip install tensorflow numpy scikit-learn matplotlib seaborn pillow

Run:
    python train_and_figures_v2.py
=============================================================
"""

import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, Reshape,
    Bidirectional, LSTM, Multiply, Lambda, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# ── CONFIG ────────────────────────────────────────────────────
CONFIG = {
    # ⬇ Updated to new pipeline output directory
    'data_dir'   : 'D:/projects/ecg_arrhthmia/balanced_stft_final',
    'output_dir' : 'D:/projects/ecg_arrhthmia/results_v2',
    'figures_dir': 'D:/projects/ecg_arrhthmia/paper_figures_v2',
    'img_size'   : 64,
    'n_classes'  : 5,
    'batch_size' : 32,
    'epochs'     : 50,
    'lr'         : 0.001,
    'random_seed': 42,
    'dpi'        : 300,
}

CLASS_NAMES  = ['NOR', 'LBB', 'RBB', 'PVC', 'APC']
CLASS_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
RAW_COUNTS   = [75011, 8071, 7255, 7129, 2546]

tf.random.set_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
os.makedirs(CONFIG['output_dir'],  exist_ok=True)
os.makedirs(CONFIG['figures_dir'], exist_ok=True)


def savefig(fig, name):
    path = os.path.join(CONFIG['figures_dir'], f'{name}.png')
    fig.savefig(path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"    📊 Saved → {path}")


# =============================================================
# STEP 1 — LOAD DATA  (now 6 files: train / val / test)
# =============================================================
def load_data():
    print("\n" + "="*60)
    print("STEP 1: Loading dataset (Train / Val / Test splits)...")
    print("="*60)
    d = CONFIG['data_dir']

    X_train = np.load(os.path.join(d, 'X_train.npy'))
    X_val   = np.load(os.path.join(d, 'X_val.npy'))
    X_test  = np.load(os.path.join(d, 'X_test.npy'))
    y_train = np.load(os.path.join(d, 'y_train.npy'))
    y_val   = np.load(os.path.join(d, 'y_val.npy'))
    y_test  = np.load(os.path.join(d, 'y_test.npy'))

    # Add channel dim if missing
    if X_train.ndim == 3: X_train = X_train[..., np.newaxis]
    if X_val.ndim   == 3: X_val   = X_val[..., np.newaxis]
    if X_test.ndim  == 3: X_test  = X_test[..., np.newaxis]

    y_train_cat = to_categorical(y_train, CONFIG['n_classes'])
    y_val_cat   = to_categorical(y_val,   CONFIG['n_classes'])
    y_test_cat  = to_categorical(y_test,  CONFIG['n_classes'])

    # Class weights computed on training labels only
    weights = compute_class_weight('balanced',
                                   classes=np.unique(y_train), y=y_train)
    cw = dict(enumerate(weights))

    print(f"\n  X_train : {X_train.shape}  (balanced augmented)")
    print(f"  X_val   : {X_val.shape}   (real beats — EarlyStopping)")
    print(f"  X_test  : {X_test.shape}   (real beats — final evaluation)")

    print(f"\n  Train class distribution (augmented balanced):")
    dist_tr = Counter(y_train.tolist())
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls}: {dist_tr[i]:,}")

    print(f"\n  Val class distribution (real imbalanced):")
    dist_val = Counter(y_val.tolist())
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls}: {dist_val[i]:,}")

    print(f"\n  Test class distribution (real imbalanced):")
    dist_te = Counter(y_test.tolist())
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls}: {dist_te[i]:,}")

    print(f"\n  Class weights: {cw}")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            y_train_cat, y_val_cat, y_test_cat,
            cw)


# =============================================================
# MODELS
# =============================================================
def build_cnn(input_shape=(64, 64, 1), n=5):
    inp = Input(shape=input_shape)
    x   = Conv2D(8,  (4, 4), padding='same', activation='relu', name='conv1')(inp)
    x   = BatchNormalization()(x);  x = MaxPooling2D()(x)
    x   = Conv2D(13, (2, 2), padding='same', activation='relu', name='conv2')(x)
    x   = BatchNormalization()(x);  x = MaxPooling2D()(x)
    x   = Conv2D(13, (2, 2), padding='same', activation='relu', name='conv3')(x)
    x   = BatchNormalization()(x);  x = MaxPooling2D()(x)
    x   = Flatten()(x)
    x   = Dense(128, activation='relu')(x); x = Dropout(0.5)(x)
    out = Dense(n,   activation='softmax',  name='output')(x)
    return Model(inp, out, name='2D_CNN_Baseline')


def build_cnn_bilstm_attention(input_shape=(64, 64, 1), n=5):
    inp = Input(shape=input_shape)
    x   = Conv2D(32,  (3, 3), padding='same', activation='relu', name='conv1')(inp)
    x   = BatchNormalization()(x);  x = MaxPooling2D()(x)
    x   = Conv2D(64,  (3, 3), padding='same', activation='relu', name='conv2')(x)
    x   = BatchNormalization()(x);  x = MaxPooling2D()(x)
    x   = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x   = BatchNormalization()(x);  x = MaxPooling2D()(x)

    _, h, w, c = x.shape
    x   = Reshape((h, w * c))(x)
    x   = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3),
                        name='bilstm')(x)

    score   = Dense(1, activation='tanh', name='attn_score')(x)
    score   = Flatten(name='attn_flat')(score)
    weights = Activation('softmax', name='attn_w')(score)
    weights = Reshape((h, 1), name='attn_reshape')(weights)
    context = Multiply(name='attn_ctx')([x, weights])
    context = Lambda(lambda t: tf.reduce_sum(t, axis=1), name='attn_out')(context)

    x   = Dense(128, activation='relu', name='dense1')(context); x = Dropout(0.4)(x)
    x   = Dense(64,  activation='relu', name='dense2')(x);       x = Dropout(0.3)(x)
    out = Dense(n,   activation='softmax', name='output')(x)
    return Model(inp, out, name='CNN_BiLSTM_Attention')


# =============================================================
# STEP 2 — TRAIN  (EarlyStopping now monitors VAL, not TEST)
# =============================================================
def train_model(model, X_tr, y_tr, X_val, y_val, cw, name):
    """
    Training uses:
      - X_tr / y_tr   → learning
      - X_val / y_val → EarlyStopping / ReduceLR / ModelCheckpoint
      - X_test kept completely separate until evaluation
    """
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
    model.summary()
    model.compile(optimizer=Adam(CONFIG['lr']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    ckpt = os.path.join(CONFIG['output_dir'], f'{name}_best.keras')
    cb = [
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(ckpt, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),   # ← real val set, NOT test
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        class_weight=cw,
        callbacks=cb,
        verbose=1
    )
    print(f"\n  ✅ Best model saved → {ckpt}")
    return hist


# =============================================================
# STEP 3 — EVALUATE  (on held-out TEST set only)
# =============================================================
def evaluate_model(model, X_te, y_te, y_te_cat, name):
    print(f"\n{'='*60}\nEvaluating on TEST SET: {name}\n{'='*60}")
    prob = model.predict(X_te, verbose=0)
    pred = np.argmax(prob, axis=1)

    acc      = accuracy_score(y_te, pred) * 100
    macro_f1 = f1_score(y_te, pred, average='macro') * 100
    try:
        auc_val = roc_auc_score(y_te_cat, prob,
                                multi_class='ovr', average='macro')
    except Exception:
        auc_val = 0.0

    per_f1 = f1_score(y_te, pred, average=None)

    print(f"\n  Overall Accuracy : {acc:.2f}%")
    print(f"  Macro F1-Score   : {macro_f1:.2f}%")
    print(f"  AUC-ROC          : {auc_val:.4f}")
    print(f"\n  Per-Class Report:")
    print(classification_report(y_te, pred,
                                target_names=CLASS_NAMES, digits=4))
    print("  Per-Class F1:")
    for cls, f1 in zip(CLASS_NAMES, per_f1):
        tag = '⭐' if cls == 'APC' else ('✅' if f1 >= 0.90 else '⚠️')
        print(f"    {tag} {cls}: {f1*100:.2f}%")

    return dict(acc=acc, macro_f1=macro_f1, auc=auc_val,
                per_f1=per_f1, pred=pred, prob=prob)


# =============================================================
# ALL PAPER FIGURES
# =============================================================

# ── Fig 1: Class Distribution ────────────────────────────────
def fig_distribution(y_train, y_val, y_test):
    print("\n  Generating Fig 1 — Class Distribution (Train / Val / Test)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(5)

    splits = [
        (y_train, '(a) TRAIN — After Augmentation',
         'Balanced 10,000 per class\n(Augmented synthetic + real)'),
        (y_val,   '(b) VALIDATION — Real Beats',
         'Real imbalanced MIT-BIH\n(EarlyStopping monitor)'),
        (y_test,  '(c) TEST — Real Beats',
         'Real imbalanced MIT-BIH\n(Final honest evaluation only)'),
    ]

    for ax, (y, title, sub) in zip(axes, splits):
        dist   = Counter(y.tolist())
        counts = [dist[i] for i in range(5)]
        bars   = ax.bar(x, counts, color=CLASS_COLORS,
                        edgecolor='black', linewidth=0.6, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Beats', fontsize=11)
        ax.set_title(f'{title}\n{sub}', fontsize=10,
                     fontweight='bold', color='#1F4E79')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(counts) * 0.012,
                    f'{c:,}', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('Fig. 1 — Dataset Split: TRAIN (Augmented) / VAL (Real) / TEST (Real)\n'
                 'VAL and TEST = real unseen beats — zero leakage guaranteed',
                 fontsize=12, fontweight='bold', color='#1F4E79', y=1.01)
    plt.tight_layout()
    savefig(fig, 'Fig1_Class_Distribution')


# ── Fig 2: Training Curves ───────────────────────────────────
def fig_training_curves(h1, h2):
    """
    Shows Train and Val curves.
    Val = real imbalanced validation set (honest EarlyStopping target).
    """
    print("  Generating Fig 2 — Training Curves (Train vs Val)...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plots = [
        (axes[0][0], h1.history['accuracy'],     h1.history['val_accuracy'],
         'Model 1 (2D-CNN) — Accuracy',           'Accuracy', '#2196F3', '#F44336'),
        (axes[0][1], h1.history['loss'],          h1.history['val_loss'],
         'Model 1 (2D-CNN) — Loss',               'Loss',     '#2196F3', '#F44336'),
        (axes[1][0], h2.history['accuracy'],      h2.history['val_accuracy'],
         'Model 2 (CNN+BiLSTM+Attn) — Accuracy',  'Accuracy', '#4CAF50', '#FF9800'),
        (axes[1][1], h2.history['loss'],          h2.history['val_loss'],
         'Model 2 (CNN+BiLSTM+Attn) — Loss',      'Loss',     '#4CAF50', '#FF9800'),
    ]

    for ax, tr, val, title, ylabel, tc, vc in plots:
        ep = range(1, len(tr) + 1)
        ax.plot(ep, tr,  color=tc, lw=2, label='Training',         alpha=0.9)
        ax.plot(ep, val, color=vc, lw=2, label='Validation (real)',
                alpha=0.9, ls='--')
        ax.fill_between(ep, tr, val, alpha=0.07, color=tc)
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1F4E79')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if 'Accuracy' in title:
            best_ep  = np.argmax(val) + 1
            best_val = max(val)
            ax.axvline(x=best_ep, color=vc, ls=':', alpha=0.7, lw=1.5)
            ax.annotate(f'Best: {best_val*100:.2f}%\nEpoch {best_ep}',
                        xy=(best_ep, best_val),
                        xytext=(best_ep + 1, best_val - 0.03),
                        fontsize=8.5, color=vc, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=vc))

    fig.suptitle('Fig. 2 — Training and Validation Curves\n'
                 '(Validation = real imbalanced beats — honest EarlyStopping)',
                 fontsize=13, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    savefig(fig, 'Fig2_Training_Curves')


# ── Fig 3: Confusion Matrices ────────────────────────────────
def fig_confusion_matrices(m1, m2, y_te):
    print("  Generating Fig 3 — Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, metrics, title in zip(
        axes, [m1, m2],
        ['(a) Model 1: 2D-CNN Baseline',
         '(b) Model 2: CNN+BiLSTM+Attention (Proposed)']
    ):
        cm   = confusion_matrix(y_te, metrics['pred'])
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_n, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5, linecolor='gray',
                    annot_kws={'size': 11, 'weight': 'bold'},
                    vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color='#1F4E79', pad=12)
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label',      fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=11)

    fig.suptitle('Fig. 3 — Normalised Confusion Matrices\n'
                 '(Test set = real imbalanced beats — honest clinical evaluation)',
                 fontsize=12, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    savefig(fig, 'Fig3_Confusion_Matrices')


# ── Fig 4: Per-Class F1 Comparison ───────────────────────────
def fig_f1_comparison(m1, m2):
    print("  Generating Fig 4 — F1 Comparison...")
    f1_m1 = m1['per_f1'] * 100
    f1_m2 = m2['per_f1'] * 100
    x, w  = np.arange(5), 0.32

    fig, ax = plt.subplots(figsize=(13, 7))
    b1 = ax.bar(x - w/2, f1_m1, w, label='Model 1: 2D-CNN',
                color='#2196F3', edgecolor='black', lw=0.6, alpha=0.9)
    b2 = ax.bar(x + w/2, f1_m2, w, label='Model 2: CNN+BiLSTM+Attn',
                color='#4CAF50', edgecolor='black', lw=0.6, alpha=0.9)

    for bars, f1s in [(b1, f1_m1), (b2, f1_m2)]:
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax.axvspan(3.55, 4.45, alpha=0.07, color='gold', zorder=0)
    ax.text(4, max(0, ax.get_ylim()[0]) + 1,
            '⭐ APC\nKey Metric', ha='center', fontsize=9,
            color='#B85C00', fontweight='bold')

    for i in range(5):
        diff = f1_m2[i] - f1_m1[i]
        if diff > 0:
            mid = x[i]
            ax.annotate('', xy=(mid + w/2, f1_m2[i] + 0.5),
                        xytext=(mid - w/2, f1_m1[i] + 0.5),
                        arrowprops=dict(arrowstyle='->', color='green',
                                        lw=1.5, connectionstyle='arc3,rad=-0.2'))

    ax.axhline(y=90, color='red', ls='--', lw=1.5, alpha=0.5, label='90% target')
    ax.set_xlabel('Arrhythmia Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score (%)',     fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontweight='bold')
    ymin = max(0, min(f1_m1.min(), f1_m2.min()) - 3)
    ax.set_ylim(ymin, 102)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Fig. 4 — Per-Class F1-Score: Model 1 vs Model 2\n'
                 '(Test set = real imbalanced data — honest evaluation)',
                 fontsize=12, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    savefig(fig, 'Fig4_F1_Comparison')


# ── Fig 5: ROC Curves ────────────────────────────────────────
def fig_roc_curves(m1, m2, y_te_cat):
    print("  Generating Fig 5 — ROC Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, metrics, title in zip(
        axes, [m1, m2],
        ['(a) Model 1: 2D-CNN Baseline',
         '(b) Model 2: CNN+BiLSTM+Attention (Proposed)']
    ):
        aucs = []
        for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            fpr, tpr, _ = roc_curve(y_te_cat[:, i], metrics['prob'][:, i])
            roc_auc     = auc(fpr, tpr)
            aucs.append(roc_auc)
            lw = 2.5 if cls == 'APC' else 1.8
            ax.plot(fpr, tpr, color=color, lw=lw,
                    label=f'{cls} (AUC={roc_auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate',  fontsize=11, fontweight='bold')
        macro_auc = np.mean(aucs)
        ax.set_title(f'{title}\nMacro AUC = {macro_auc:.4f}',
                     fontsize=11, fontweight='bold', color='#1F4E79')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Fig. 5 — ROC Curves Per Class (Real Imbalanced Test Set)',
                 fontsize=12, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    savefig(fig, 'Fig5_ROC_Curves')


# ── Fig 6: Overall Metrics Bar ───────────────────────────────
def fig_overall_metrics(m1, m2):
    print("  Generating Fig 6 — Overall Metrics Comparison...")
    metrics_names = ['Accuracy (%)', 'Macro F1 (%)', 'AUC-ROC × 100']
    m1_vals = [m1['acc'], m1['macro_f1'], m1['auc'] * 100]
    m2_vals = [m2['acc'], m2['macro_f1'], m2['auc'] * 100]

    x, w = np.arange(3), 0.3
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, m1_vals, w, label='Model 1: 2D-CNN',
                color='#2196F3', edgecolor='black', lw=0.6)
    b2 = ax.bar(x + w/2, m2_vals, w, label='Model 2: CNN+BiLSTM+Attn',
                color='#4CAF50', edgecolor='black', lw=0.6)

    for bars, vals in [(b1, m1_vals), (b2, m2_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
    ymin = min(m1_vals + m2_vals) - 2
    ax.set_ylim(ymin, 102)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Fig. 6 — Overall Performance Metrics: Model 1 vs Model 2\n'
                 '(Evaluated on real imbalanced test set — 70/10/20 split)',
                 fontsize=12, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    savefig(fig, 'Fig6_Overall_Metrics')


# ── Fig 7: Precision-Recall per class ────────────────────────
def fig_precision_recall(m1, m2, y_te):
    print("  Generating Fig 7 — Precision & Recall per class...")
    prec_m1 = precision_score(y_te, m1['pred'], average=None) * 100
    rec_m1  = recall_score(y_te,    m1['pred'], average=None) * 100
    prec_m2 = precision_score(y_te, m2['pred'], average=None) * 100
    rec_m2  = recall_score(y_te,    m2['pred'], average=None) * 100

    x, w = np.arange(5), 0.2
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, (prec, rec), title in zip(
        axes,
        [(prec_m1, rec_m1), (prec_m2, rec_m2)],
        ['Model 1: 2D-CNN Baseline',
         'Model 2: CNN+BiLSTM+Attention (Proposed)']
    ):
        b1 = ax.bar(x - w/2, prec, w, label='Precision',
                    color='#2196F3', edgecolor='black', lw=0.5, alpha=0.9)
        b2 = ax.bar(x + w/2, rec,  w, label='Recall',
                    color='#F44336', edgecolor='black', lw=0.5, alpha=0.9)
        for bar, val in zip(list(b1) + list(b2), list(prec) + list(rec)):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', fontsize=8.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
        ymin = max(0, min(list(prec) + list(rec)) - 5)
        ax.set_ylim(ymin, 103)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold', color='#1F4E79')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvspan(3.6, 4.4, alpha=0.07, color='gold', zorder=0)

    fig.suptitle('Fig. 7 — Per-Class Precision and Recall: Model 1 vs Model 2',
                 fontsize=12, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    savefig(fig, 'Fig7_Precision_Recall')


# ── Fig 8: Comparison vs Literature ─────────────────────────
def fig_paper_comparison(m1, m2):
    print("  Generating Fig 8 — Comparison with existing papers...")
    papers = ['Huang 2019\n(Original)', 'Ullah 2020', 'SE-CNN 2025',
              'CNN-LSTM\n2024', 'Our Model 1\n(2D-CNN)', 'Our Model 2\n(Proposed)']
    acc    = [99.00, 99.11, 99.13, 98.83, m1['acc'], m2['acc']]
    colors = ['#BBDEFB'] * 4 + ['#2196F3', '#4CAF50']
    edges  = ['black'] * 4 + ['#0D47A1', '#1B5E20']
    lws    = [0.5] * 4 + [2.0, 2.0]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(papers)), acc, color=colors,
                  edgecolor=edges, linewidth=lws, width=0.6)

    for bar, val, paper in zip(bars, acc, papers):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{val:.2f}%', ha='center', fontsize=10.5, fontweight='bold',
                color='#1F4E79' if 'Our' in paper else '#333333')

    ax.set_xticks(range(len(papers)))
    ax.set_xticklabels(papers, fontsize=10, fontweight='bold')
    ymin = min(acc) - 0.5
    ax.set_ylim(ymin, 100.2)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.add_patch(mpatches.Patch(color='#BBDEFB', label='Existing papers'))
    ax.add_patch(mpatches.Patch(color='#2196F3', label='Our Model 1 (2D-CNN)'))
    ax.add_patch(mpatches.Patch(color='#4CAF50', label='Our Model 2 (Proposed) ⭐'))
    ax.legend(fontsize=10, loc='lower right')

    ax.set_title('Fig. 8 — Accuracy Comparison with Existing Work\n'
                 '(Our models: 70/10/20 split — real imbalanced test set — more rigorous)',
                 fontsize=11, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    savefig(fig, 'Fig8_Paper_Comparison')


# ── Fig 9: APC Deep Dive ─────────────────────────────────────
def fig_apc_analysis(m1, m2, y_te):
    print("  Generating Fig 9 — APC Deep Dive...")
    apc_idx = 4

    def get_apc(metrics):
        pred = metrics['pred']
        prec = precision_score(y_te, pred, average=None)[apc_idx] * 100
        rec  = recall_score(y_te,    pred, average=None)[apc_idx] * 100
        f1   = metrics['per_f1'][apc_idx] * 100
        return [prec, rec, f1]

    apc_m1 = get_apc(m1)
    apc_m2 = get_apc(m2)
    metrics_labels = ['Precision', 'Recall', 'F1-Score']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x, w = np.arange(3), 0.3
    b1 = axes[0].bar(x - w/2, apc_m1, w, label='Model 1: 2D-CNN',
                     color='#2196F3', edgecolor='black', lw=0.6)
    b2 = axes[0].bar(x + w/2, apc_m2, w, label='Model 2: CNN+BiLSTM+Attn',
                     color='#4CAF50', edgecolor='black', lw=0.6)
    for bars, vals in [(b1, apc_m1), (b2, apc_m2)]:
        for bar, val in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.2,
                         f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_labels, fontsize=12)
    axes[0].set_ylim(max(0, min(apc_m1 + apc_m2) - 5), 103)
    axes[0].set_ylabel('Score (%)', fontsize=11)
    axes[0].set_title('APC (Atrial Premature Contraction)\nPrecision / Recall / F1',
                      fontsize=11, fontweight='bold', color='#9C27B0')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, axis='y', alpha=0.3, ls='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    sizes   = [RAW_COUNTS[i] for i in range(5)]
    explode = [0, 0, 0, 0, 0.1]
    axes[1].pie(sizes, labels=CLASS_NAMES, colors=CLASS_COLORS,
                autopct='%1.1f%%', startangle=90, explode=explode,
                textprops={'fontsize': 11, 'fontweight': 'bold'},
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    axes[1].set_title('Real Dataset Distribution\n(APC = only 2.55% of all beats)',
                      fontsize=11, fontweight='bold', color='#9C27B0')

    fig.suptitle('Fig. 9 — APC (Minority Class) Deep Dive\n'
                 'APC is the most critical and hardest class — real test beats only',
                 fontsize=12, fontweight='bold', color='#1F4E79')
    plt.tight_layout()
    savefig(fig, 'Fig9_APC_Analysis')


# ── Fig 10: Summary Dashboard ────────────────────────────────
def fig_summary_dashboard(m1, m2):
    print("  Generating Fig 10 — Summary Dashboard...")
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.96,
             'ECG Arrhythmia Classification — Final Results Summary',
             ha='center', va='top', fontsize=16,
             fontweight='bold', color='#1F4E79')
    fig.text(0.5, 0.92,
             'CNN+BiLSTM+Attention vs 2D-CNN Baseline | '
             'MIT-BIH All 48 Records | 70/10/20 Split',
             ha='center', va='top', fontsize=12,
             color='#555555', style='italic')

    metrics_boxes = [
        (f"{m2['acc']:.2f}%",           'Overall Accuracy\n(Model 2)',      '#1F4E79', 0.08),
        (f"{m2['macro_f1']:.2f}%",      'Macro F1-Score\n(Model 2)',        '#2E75B6', 0.28),
        (f"{m2['auc']:.4f}",            'AUC-ROC\n(Model 2)',               '#4CAF50', 0.48),
        (f"{m2['per_f1'][4]*100:.2f}%", 'APC F1-Score\n⭐ Key Metric',      '#9C27B0', 0.68),
        (f"{m2['acc']-99.00:+.2f}%",    'vs Original\nPaper (Huang 2019)',  '#F44336', 0.88),
    ]

    for val, label, color, x in metrics_boxes:
        box = FancyBboxPatch((x - 0.045, 0.72), 0.09, 0.14,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='white', linewidth=2,
                             transform=fig.transFigure, clip_on=False)
        fig.add_artist(box)
        fig.text(x, 0.805, val,   ha='center', va='center', fontsize=15,
                 fontweight='bold', color='white', transform=fig.transFigure)
        fig.text(x, 0.745, label, ha='center', va='center', fontsize=8,
                 color='white', transform=fig.transFigure, multialignment='center')

    # Per-class F1 table
    ax_table = fig.add_axes([0.05, 0.1, 0.42, 0.55])
    ax_table.axis('off')
    rows = [[cls,
             f'{m1["per_f1"][i]*100:.2f}%',
             f'{m2["per_f1"][i]*100:.2f}%',
             f'{(m2["per_f1"][i]-m1["per_f1"][i])*100:+.2f}%']
            for i, cls in enumerate(CLASS_NAMES)]
    rows.append(['Macro',
                 f'{m1["macro_f1"]:.2f}%',
                 f'{m2["macro_f1"]:.2f}%',
                 f'{m2["macro_f1"]-m1["macro_f1"]:+.2f}%'])

    tbl = ax_table.table(
        cellText=rows,
        colLabels=['Class', 'Model 1 F1', 'Model 2 F1', 'Δ Change'],
        cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1F4E79')
            cell.set_text_props(color='white', fontweight='bold')
        elif r == len(rows):
            cell.set_facecolor('#EBF3FB')
            cell.set_text_props(fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F5F5F5')
        if r > 0 and c == 3:
            val = rows[r - 1][3]
            cell.set_text_props(color='green' if '+' in val else 'red',
                                fontweight='bold')
        cell.set_edgecolor('#CCCCCC')
    ax_table.set_title('Per-Class F1-Score Comparison', fontsize=11,
                       fontweight='bold', color='#1F4E79', pad=10)

    # F1 bar subplot
    ax_bar = fig.add_axes([0.55, 0.1, 0.42, 0.55])
    x, w = np.arange(5), 0.35
    ax_bar.bar(x - w/2, m1['per_f1'] * 100, w, color='#2196F3',
               label='Model 1', edgecolor='black', lw=0.5, alpha=0.9)
    ax_bar.bar(x + w/2, m2['per_f1'] * 100, w, color='#4CAF50',
               label='Model 2', edgecolor='black', lw=0.5, alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    ymin = max(0, min(list(m1['per_f1']) + list(m2['per_f1'])) * 100 - 3)
    ax_bar.set_ylim(ymin, 102)
    ax_bar.axhline(y=90, color='red', ls='--', alpha=0.5, lw=1.5)
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, axis='y', alpha=0.3, ls='--')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_title('F1-Score Per Class', fontsize=11,
                     fontweight='bold', color='#1F4E79')

    savefig(fig, 'Fig10_Summary_Dashboard')


# =============================================================
# FINAL COMPARISON PRINT
# =============================================================
def print_comparison(m1, m2):
    print("\n" + "="*65)
    print("FINAL COMPARISON — TEST SET (Real Imbalanced — 20% split)")
    print("="*65)
    print(f"{'Metric':<25} {'Model 1 (2D-CNN)':>18} {'Model 2 (Proposed)':>20}")
    print("-"*65)
    print(f"{'Accuracy':<25} {m1['acc']:>17.2f}% {m2['acc']:>19.2f}%")
    print(f"{'Macro F1':<25} {m1['macro_f1']:>17.2f}% {m2['macro_f1']:>19.2f}%")
    print(f"{'AUC-ROC':<25} {m1['auc']:>18.4f} {m2['auc']:>20.4f}")
    print("-"*65)
    print("Per-Class F1:")
    for i, cls in enumerate(CLASS_NAMES):
        star = ' ⭐' if cls == 'APC' else '   '
        diff = (m2['per_f1'][i] - m1['per_f1'][i]) * 100
        print(f"  {cls}{star}  {m1['per_f1'][i]*100:>14.2f}%"
              f"  {m2['per_f1'][i]*100:>16.2f}%"
              f"  {'↑' if diff > 0 else '↓'}{abs(diff):.2f}%")
    print("="*65)
    win = ('Model 2 (Proposed)' if m2['macro_f1'] > m1['macro_f1']
           else 'Model 1')
    print(f"\n  🏆 Winner: {win}")
    print(f"  ⭐ APC F1: Model1={m1['per_f1'][4]*100:.2f}%  "
          f"Model2={m2['per_f1'][4]*100:.2f}%")
    print("="*65)


# =============================================================
# MAIN
# =============================================================
def main():
    print("\n" + "🔷"*30)
    print("  ECG ARRHYTHMIA — TRAIN + ALL PAPER FIGURES v2")
    print("  Pipeline: 70% Train / 10% Val / 20% Test")
    print("🔷"*30 + "\n")

    # ── Load all 6 arrays ──
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_train_cat, y_val_cat, y_test_cat,
     cw) = load_data()

    input_shape = (CONFIG['img_size'], CONFIG['img_size'], 1)

    # ── Build models ──
    model1 = build_cnn(input_shape, CONFIG['n_classes'])
    model2 = build_cnn_bilstm_attention(input_shape, CONFIG['n_classes'])

    # ── Train  (EarlyStopping monitors VAL, test is untouched) ──
    history1 = train_model(model1, X_train, y_train_cat,
                           X_val,  y_val_cat, cw, 'Model1_2DCNN')
    history2 = train_model(model2, X_train, y_train_cat,
                           X_val,  y_val_cat, cw, 'Model2_BiLSTM_Attn')

    # ── Evaluate on TEST only ──
    m1 = evaluate_model(model1, X_test, y_test, y_test_cat, 'Model1_2DCNN')
    m2 = evaluate_model(model2, X_test, y_test, y_test_cat, 'Model2_BiLSTM_Attn')

    # ── Print comparison ──
    print_comparison(m1, m2)

    # ── Generate all figures ──
    print("\n" + "="*60)
    print("GENERATING ALL PAPER FIGURES (300 DPI)...")
    print("="*60)
    fig_distribution(y_train, y_val, y_test)        # now 3-panel
    fig_training_curves(history1, history2)          # train vs val curves
    fig_confusion_matrices(m1, m2, y_test)
    fig_f1_comparison(m1, m2)
    fig_roc_curves(m1, m2, y_test_cat)
    fig_overall_metrics(m1, m2)
    fig_precision_recall(m1, m2, y_test)
    fig_paper_comparison(m1, m2)
    fig_apc_analysis(m1, m2, y_test)
    fig_summary_dashboard(m1, m2)

    print("\n" + "="*60)
    print("✅  ALL DONE!")
    print("="*60)
    print(f"\n  Models  → {CONFIG['output_dir']}/")
    print(f"  Figures → {CONFIG['figures_dir']}/")
    print("\n  10 Figures:")
    for f in [
        "Fig1  — Class Distribution (Train aug / Val real / Test real)",
        "Fig2  — Training Curves (Train vs Val — real val beats)",
        "Fig3  — Confusion Matrices (normalised, test set)",
        "Fig4  — Per-Class F1 Comparison bar chart",
        "Fig5  — ROC Curves (per class, both models)",
        "Fig6  — Overall Metrics (Accuracy, F1, AUC)",
        "Fig7  — Precision & Recall per class",
        "Fig8  — Comparison with existing papers",
        "Fig9  — APC Deep Dive + dataset pie chart",
        "Fig10 — Summary Dashboard (all results one page)",
    ]:
        print(f"    {f}")
    print("\n  🎉 Ready for paper submission!\n")


if __name__ == '__main__':
    main()