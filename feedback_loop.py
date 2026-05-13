import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from real_data_loader import load_real_data, prepare_real_features, get_feature_columns
import os
import json
from datetime import datetime

FEEDBACK_LOG = 'data/feedback_log.json'

def load_feedback_log():
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, 'r') as f:
            return json.load(f)
    return []

def save_feedback(feedback_entry):
    log = load_feedback_log()
    log.insert(0, feedback_entry)
    log = log[:200]  # keep last 200
    with open(FEEDBACK_LOG, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Feedback saved. Total feedback collected: {len(log)}")

def retrain_on_feedback(X_new, y_new, model_type='false_decline'):
    """
    Online learning — model retrains on analyst feedback.
    Uses SGDClassifier which supports partial_fit (incremental learning).
    Gets smarter every time an analyst corrects it.
    """
    print(f"Retraining {model_type} model on new feedback...")

    sgd_path = f'models/sgd_{model_type}.pkl'

    if os.path.exists(sgd_path):
        sgd_model = joblib.load(sgd_path)
    else:
        sgd_model = SGDClassifier(
            loss='log_loss',
            random_state=42,
            max_iter=1000
        )

    sgd_model.partial_fit(X_new, y_new, classes=[0, 1])
    joblib.dump(sgd_model, sgd_path)
    print(f"✅ Model retrained and saved to {sgd_path}")
    return sgd_model


def process_feedback(transaction_features, true_label, fraud_prob, fd_prob, decision):
    """
    Called when analyst clicks correct/wrong on dashboard.
    Logs the feedback and triggers retraining.
    """
    feedback_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'true_label': int(true_label),
        'fraud_prob': round(float(fraud_prob), 4),
        'fd_prob': round(float(fd_prob), 4),
        'original_decision': decision,
        'was_correct': True if (
            (decision == 'AUTO DECLINE' and true_label == 1) or
            (decision == 'AUTO APPROVE' and true_label != 1)
        ) else False
    }
    save_feedback(feedback_entry)

    # Retrain on this single new example
    feature_cols = get_feature_columns()
    X_new = np.array(transaction_features).reshape(1, -1)
    y_new = np.array([1 if true_label == 2 else 0])

    retrain_on_feedback(X_new, y_new)
    return feedback_entry


def get_feedback_stats():
    log = load_feedback_log()
    if not log:
        return {'total': 0, 'correct': 0, 'accuracy': 0}
    total = len(log)
    correct = sum(1 for f in log if f['was_correct'])
    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': round(correct / total * 100, 1)
    }


if __name__ == "__main__":
    print("Initializing feedback loop with real data...")

    df = load_real_data()
    df = prepare_real_features(df)
    feature_cols = get_feature_columns()

    # Warm start — pretrain SGD on small sample
    sample = df.sample(1000, random_state=42)
    X = sample[feature_cols].fillna(0).values
    y = (sample['label'] == 2).astype(int).values

    sgd = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000)
    sgd.partial_fit(X, y, classes=[0, 1])
    joblib.dump(sgd, 'models/sgd_false_decline.pkl')

    print("✅ Feedback loop initialized!")
    print("SGD model warm-started on 1000 samples")
    print("Ready to learn from analyst corrections in real time")

    # Show stats
    stats = get_feedback_stats()
    print(f"\nFeedback log stats: {stats}")
    