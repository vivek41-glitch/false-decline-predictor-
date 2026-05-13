import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

from real_data_loader import load_real_data, prepare_real_features, get_feature_columns


if __name__ == "__main__":

    # ── LOAD REAL DATA ────────────────────────────────────────
    df = load_real_data()
    df = prepare_real_features(df)
    feature_cols = get_feature_columns()

    X = df[feature_cols].fillna(0)
    y = df['label']

    y_fraud = (y == 1).astype(int)
    y_fd = (y == 2).astype(int)

    # ── TRAIN TEST SPLIT ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    _, _, y_train_fraud, y_test_fraud = train_test_split(
        X, y_fraud, test_size=0.2, random_state=42, stratify=y
    )
    _, _, y_train_fd, y_test_fd = train_test_split(
        X, y_fd, test_size=0.2, random_state=42, stratify=y
    )

    # ── MODEL 1: FRAUD DETECTOR ───────────────────────────────
    print("\n--- Training Fraud Detector (XGBoost) on REAL data ---")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train_fraud)

    fraud_model = XGBClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.1, random_state=42,
        eval_metric='logloss', verbosity=0
    )
    fraud_model.fit(X_res, y_res)
    y_pred = fraud_model.predict(X_test)
    y_prob = fraud_model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test_fraud, y_pred, target_names=['Legit', 'Fraud']))
    print(f"AUC-ROC: {roc_auc_score(y_test_fraud, y_prob):.4f}")

    # ── MODEL 2: FALSE DECLINE DETECTOR ──────────────────────
    print("\n--- Training False Decline Detector (Random Forest) on REAL data ---")
    X_res2, y_res2 = sm.fit_resample(X_train, y_train_fd)

    fd_model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, n_jobs=-1
    )
    fd_model.fit(X_res2, y_res2)
    y_pred2 = fd_model.predict(X_test)
    y_prob2 = fd_model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test_fd, y_pred2, target_names=['Not FD', 'False Decline']))
    print(f"AUC-ROC: {roc_auc_score(y_test_fd, y_prob2):.4f}")

    # ── MODEL 3: ANOMALY DETECTOR ─────────────────────────────
    print("\n--- Training Anomaly Detector (Isolation Forest) ---")
    anomaly_model = IsolationForest(
        n_estimators=100, contamination=0.02,
        random_state=42
    )
    anomaly_model.fit(X_train)
    print("Isolation Forest trained successfully")

    # ── SAVE MODELS ───────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(fraud_model, 'models/fraud_model.pkl')
    joblib.dump(fd_model, 'models/false_decline_model.pkl')
    joblib.dump(anomaly_model, 'models/anomaly_model.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    print("\n✅ All models trained on REAL data and saved!")

    # ── BUSINESS IMPACT ───────────────────────────────────────
    print("\n--- Business Impact on REAL data ---")
    caught = sum((y_pred2 == 1) & (y_test_fd == 1))
    total_fd = sum(y_test_fd == 1)
    avg_txn = df['Amount'].mean()
    print(f"False declines in test set: {total_fd:,}")
    print(f"False declines caught: {caught:,}")
    print(f"Catch rate: {caught/total_fd*100:.1f}%")
    print(f"Estimated revenue saved: ${caught * avg_txn:,.2f}")
    