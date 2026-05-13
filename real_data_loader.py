import pandas as pd
import numpy as np

def load_real_data():
    """
    Load the real Kaggle Credit Card Fraud dataset.
    284,807 real transactions with 492 fraud cases.
    Features V1-V28 are PCA-transformed real transaction features.
    'Amount' is real transaction amount, 'Class' is 0=legit, 1=fraud.
    """
    print("Loading real Kaggle credit card dataset...")
    df = pd.read_csv('data/creditcard.csv')
    print(f"Loaded {len(df):,} real transactions")
    print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"Legitimate: {(df['Class']==0).sum():,}")
    return df


def prepare_real_features(df):
    """
    Prepare features from real dataset.
    V1-V28 are already PCA features from real transaction data.
    We add extra engineered features on top.
    """
    print("Preparing features from real data...")
    df = df.copy()

    # ── AMOUNT FEATURES ───────────────────────────────────────
    df['amount_log'] = np.log1p(df['Amount'])
    df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    df['is_high_amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['Amount'] < df['Amount'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['Amount'] % 10 == 0).astype(int)

    # ── TIME FEATURES ─────────────────────────────────────────
    df['hour'] = (df['Time'] % 86400) // 3600
    df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['time_normalized'] = df['Time'] / df['Time'].max()

    # ── INTERACTION FEATURES ──────────────────────────────────
    df['amount_time_interaction'] = df['amount_log'] * df['time_normalized']
    df['v1_v2_interaction'] = df['V1'] * df['V2']
    df['v3_v4_interaction'] = df['V3'] * df['V4']

    # ── VELOCITY FEATURES ─────────────────────────────────────
    df = df.sort_values('Time').reset_index(drop=True)
    df['time_hour_bucket'] = (df['Time'] // 3600).astype(int)
    df['txn_velocity_1hr'] = df.groupby('time_hour_bucket').cumcount()
    df['amount_velocity_1hr'] = df.groupby('time_hour_bucket')['Amount'].cumsum()
    df['is_high_velocity'] = (df['txn_velocity_1hr'] > df['txn_velocity_1hr'].quantile(0.95)).astype(int)

    # ── FALSE DECLINE SIMULATION ──────────────────────────────
    legitimate = df['Class'] == 0
    unusual_amount = df['is_high_amount'] == 1
    unusual_time = df['is_late_night'] == 1
    unusual_velocity = df['is_high_velocity'] == 1
    false_decline_mask = legitimate & (unusual_amount | unusual_time | unusual_velocity)

    df['false_decline_candidate'] = 0
    df.loc[false_decline_mask, 'false_decline_candidate'] = 1

    # Combined label: 0=normal, 1=fraud, 2=false decline candidate
    df['label'] = df['Class']
    df.loc[false_decline_mask & (df['Class'] == 0), 'label'] = 2

    print(f"Normal transactions: {(df['label']==0).sum():,}")
    print(f"Fraud transactions: {(df['label']==1).sum():,}")
    print(f"False decline candidates: {(df['label']==2).sum():,}")

    return df


def get_feature_columns():
    """Return the feature columns to use for training."""
    v_features = [f'V{i}' for i in range(1, 29)]
    engineered = [
        'amount_log', 'amount_zscore', 'is_high_amount',
        'is_low_amount', 'is_round_amount', 'hour',
        'is_late_night', 'is_business_hours', 'time_normalized',
        'amount_time_interaction', 'v1_v2_interaction', 'v3_v4_interaction',
        'txn_velocity_1hr', 'amount_velocity_1hr', 'is_high_velocity'
    ]
    return v_features + engineered


if __name__ == "__main__":
    df = load_real_data()
    df_prepared = prepare_real_features(df)
    feature_cols = get_feature_columns()

    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Feature list: {feature_cols}")

    df_prepared.to_csv('data/real_featured_transactions.csv', index=False)
    print("\nSaved to data/real_featured_transactions.csv")
    