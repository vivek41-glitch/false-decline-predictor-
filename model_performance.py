import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import joblib
import os
from real_data_loader import load_real_data, prepare_real_features, get_feature_columns
from sklearn.model_selection import train_test_split

def generate_performance_plots():
    print("Generating model performance plots...")

    # Load data and models
    df = load_real_data()
    df = prepare_real_features(df)
    feature_cols = get_feature_columns()

    X = df[feature_cols].fillna(0)
    y = df['label']
    y_fraud = (y == 1).astype(int)
    y_fd = (y == 2).astype(int)

    _, X_test, _, y_test_fraud = train_test_split(
        X, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
    )
    _, _, _, y_test_fd = train_test_split(
        X, y_fd, test_size=0.2, random_state=42, stratify=y_fd
    )

    fraud_model = joblib.load('models/fraud_model.pkl')
    fd_model = joblib.load('models/false_decline_model.pkl')

    os.makedirs('data/perf_plots', exist_ok=True)

    for model, y_test, name in [
        (fraud_model, y_test_fraud, 'fraud'),
        (fd_model, y_test_fd, 'false_decline')
    ]:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # ── CONFUSION MATRIX ──────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{name.replace("_"," ").title()} Model Performance', fontsize=14)

        cm = confusion_matrix(y_test, y_pred)
        im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]),
                           ha='center', va='center',
                           color='white' if cm[i, j] > cm.max()/2 else 'black',
                           fontsize=14)

        # ── ROC CURVE ─────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'AUC = {roc_auc:.4f}')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)

        # ── PRECISION RECALL CURVE ────────────────────────────
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        axes[2].plot(recall, precision, color='green', lw=2)
        axes[2].set_title('Precision-Recall Curve')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'data/perf_plots/{name}_performance.png',
                   dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved: data/perf_plots/{name}_performance.png")

    print("✅ Performance plots generated!")


if __name__ == "__main__":
    generate_performance_plots()
    