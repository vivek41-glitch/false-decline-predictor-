import pandas as pd
import numpy as np


def engineer_features(merged_df):
    """
    Create 30+ features by comparing each transaction
    against the customer's own historical profile.
    This is the heart of the false decline detector.
    """
    print("Engineering features...")
    df = merged_df.copy()

    # ── AMOUNT FEATURES ──────────────────────────────────────
    # How different is this amount vs customer's normal spending?
    df['amount_vs_avg_ratio'] = df['amount'] / (df['profile_avg_amount'] + 1)
    df['amount_vs_max_ratio'] = df['amount'] / (df['profile_max_amount'] + 1)
    df['amount_deviation'] = abs(df['amount'] - df['profile_avg_amount'])
    df['amount_zscore'] = (
        (df['amount'] - df['profile_avg_amount']) /
        (df['profile_std_amount'] + 1)
    )
    df['is_amount_unusually_high'] = (df['amount_zscore'] > 2).astype(int)
    df['is_amount_unusually_low'] = (df['amount_zscore'] < -2).astype(int)

    # ── TIME FEATURES ─────────────────────────────────────────
    # Is this transaction at an unusual time for this customer?
    df['hour_deviation'] = abs(df['hour'] - df['profile_avg_hour'])
    df['is_unusual_hour'] = (df['hour_deviation'] > 2 * df['profile_std_hour'] + 1).astype(int)
    df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # ── LOCATION FEATURES ─────────────────────────────────────
    # Is this transaction from an unusual location?
    df['is_foreign_unusual'] = (
        (df['is_foreign_country'] == 1) &
        (df['profile_foreign_rate'] < 0.1)
    ).astype(int)

    # ── MERCHANT FEATURES ─────────────────────────────────────
    # Is this a new merchant for this customer?
    df['is_new_merchant_unusual'] = (
        (df['is_new_merchant'] == 1) &
        (df['profile_new_merchant_rate'] < 0.1)
    ).astype(int)

    # Is this their favorite category or not?
    df['is_favorite_category'] = (
        df['merchant_category'] == df['profile_favorite_category']
    ).astype(int)

    df['is_unusual_category'] = (
        (df['merchant_category'].isin(['electronics', 'travel'])) &
        (df['profile_favorite_category'].isin(['grocery', 'restaurant', 'fuel']))
    ).astype(int)

    # ── CHANNEL FEATURES ──────────────────────────────────────
    df['is_online'] = (df['channel'] == 'online').astype(int)
    df['is_online_unusual'] = (
        (df['channel'] == 'online') &
        (df['profile_online_rate'] < 0.2)
    ).astype(int)

    # ── COMBINED RISK SIGNALS ─────────────────────────────────
    # Count how many suspicious signals this transaction has
    df['suspicious_signal_count'] = (
        df['is_amount_unusually_high'] +
        df['is_unusual_hour'] +
        df['is_late_night'] +
        df['is_foreign_unusual'] +
        df['is_new_merchant_unusual'] +
        df['is_unusual_category'] +
        df['is_online_unusual']
    )

    # ── FALSE DECLINE INDICATORS ──────────────────────────────
    # These are signals that suggest REAL customer despite looking suspicious
    df['legit_signal_count'] = (
        df['is_favorite_category'] +
        (df['profile_total_transactions'] > 20).astype(int) +  # long history
        (df['amount_vs_avg_ratio'] < 3).astype(int) +          # not insanely high
        (df['is_weekend']).astype(int)                          # weekends = real people shop
    )

    # ── FINAL FEATURE LIST ────────────────────────────────────
    feature_cols = [
        'amount', 'hour', 'day_of_week', 'month',
        'is_foreign_country', 'is_new_merchant', 'is_online',
        'profile_avg_amount', 'profile_std_amount',
        'profile_avg_hour', 'profile_std_hour',
        'profile_foreign_rate', 'profile_new_merchant_rate',
        'profile_online_rate', 'profile_total_transactions',
        'amount_vs_avg_ratio', 'amount_vs_max_ratio',
        'amount_deviation', 'amount_zscore',
        'is_amount_unusually_high', 'is_amount_unusually_low',
        'hour_deviation', 'is_unusual_hour', 'is_late_night', 'is_weekend',
        'is_foreign_unusual', 'is_new_merchant_unusual',
        'is_favorite_category', 'is_unusual_category',
        'is_online_unusual', 'suspicious_signal_count', 'legit_signal_count'
    ]

    print(f"Total features engineered: {len(feature_cols)}")
    return df, feature_cols


if __name__ == "__main__":
    merged_df = pd.read_csv('data/merged_transactions.csv')

    df_featured, feature_cols = engineer_features(merged_df)
    df_featured.to_csv('data/featured_transactions.csv', index=False)

    print(f"\nFeature sample:")
    print(df_featured[feature_cols].head(3).to_string())
    print(f"\nSaved to data/featured_transactions.csv")
    print(f"\nLabel distribution:")
    print(df_featured['label'].value_counts())