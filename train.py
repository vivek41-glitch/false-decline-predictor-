import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

from feature_engineering import engineer_features
from customer_profiler import build_customer_profiles, merge_profiles_with_transactions


def train_baseline_fraud_model(X_train, y_train_fraud, X_test, y_test_fraud):
    """
    Step 1: Train a basic fraud detector.
    This represents Amex's current system that makes mistakes.
    """
    print("\n--- Training Baseline Fraud Model (XGBoost) ---")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train_fraud)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Baseline Fraud Model Results:")
    print(classification_report(y_test_fraud, y_pred, target_names=['Legit', 'Fraud']))
    print(f"AUC-ROC: {roc_auc_score(y_test_fraud, y_prob):.4f}")

    return model


def train_false_decline_model(X_train, y_train_fd, X_test, y_test_fd):
    """
    Step 2: Train the false decline detector.
    This is our main innovation — catching legitimate transactions
    that look suspicious to the baseline model.
    """
    print("\n--- Training False Decline Detector (Random Forest) ---")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train_fd)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("False Decline Detector Results:")
    print(classification_report(y_test_fd, y_pred, target_names=['Not False Decline', 'False Decline']))
    print(f"AUC-ROC: {roc_auc_score(y_test_fd, y_prob):.4f}")

    return model


def train_anomaly_detector(X_train):
    """
    Step 3: Unsupervised anomaly detection.
    Catches unusual patterns without needing labels.
    """
    print("\n--- Training Anomaly Detector (Isolation Forest) ---")

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_train)
    print("Isolation Forest trained successfully")
    return model


def get_confidence_tier(fraud_prob, false_decline_prob, anomaly_score):
    """
    3-tier decision system:
    - AUTO APPROVE: low fraud risk OR high false decline probability
    - HUMAN REVIEW: uncertain cases
    - AUTO DECLINE: high fraud risk AND low false decline probability
    """
    if fraud_prob < 0.3 or false_decline_prob > 0.6:
        return "AUTO APPROVE", "green"
    elif fraud_prob > 0.7 and false_decline_prob < 0.3:
        return "AUTO DECLINE", "red"
    else:
        return "HUMAN REVIEW", "orange"


if __name__ == "__main__":
    # ── LOAD DATA ─────────────────────────────────────────────
    print("Loading data...")
    transactions_df = pd.read_csv('data/transactions.csv')
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])

    # ── BUILD PROFILES ────────────────────────────────────────
    profiles_df = build_customer_profiles(transactions_df)
    merged_df = merge_profiles_with_transactions(transactions_df, profiles_df)

    # ── ENGINEER FEATURES ─────────────────────────────────────
    df_featured, feature_cols = engineer_features(merged_df)

    X = df_featured[feature_cols].fillna(0)
    y = df_featured['label']

    # ── PREPARE LABELS ────────────────────────────────────────
    # For fraud model: 1=fraud, 0=everything else
    y_fraud = (y == 1).astype(int)

    # For false decline model: 1=false decline candidate, 0=everything else
    y_false_decline = (y == 2).astype(int)

    # ── TRAIN TEST SPLIT ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    _, _, y_train_fraud, y_test_fraud = train_test_split(
        X, y_fraud, test_size=0.2, random_state=42, stratify=y
    )
    _, _, y_train_fd, y_test_fd = train_test_split(
        X, y_false_decline, test_size=0.2, random_state=42, stratify=y
    )

    # ── TRAIN ALL 3 MODELS ────────────────────────────────────
    fraud_model = train_baseline_fraud_model(X_train, y_train_fraud, X_test, y_test_fraud)
    false_decline_model = train_false_decline_model(X_train, y_train_fd, X_test, y_test_fd)
    anomaly_model = train_anomaly_detector(X_train)

    # ── SAVE MODELS ───────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(fraud_model, 'models/fraud_model.pkl')
    joblib.dump(false_decline_model, 'models/false_decline_model.pkl')
    joblib.dump(anomaly_model, 'models/anomaly_model.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')

    print("\n✅ All 3 models trained and saved to models/")

    # ── BUSINESS IMPACT SUMMARY ───────────────────────────────
    print("\n--- Business Impact Summary ---")
    y_fd_pred = false_decline_model.predict(X_test)
    caught = sum((y_fd_pred == 1) & (y_test_fd == 1))
    total_fd = sum(y_test_fd == 1)
    avg_txn = df_featured['amount'].mean()

    print(f"False declines in test set: {total_fd}")
    print(f"False declines caught by our model: {caught}")
    print(f"Catch rate: {caught/total_fd*100:.1f}%")
    print(f"Estimated revenue saved: ${caught * avg_txn:,.2f}")
    print(f"\nWithout this system, Amex loses ~${total_fd * avg_txn:,.2f} per period in false declines")