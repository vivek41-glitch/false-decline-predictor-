import pandas as pd
import numpy as np

def build_customer_profiles(transactions_df):
    """
    For each customer, build their personal spending history.
    This is the core of our system — we compare every new transaction
    against the customer's OWN behavior, not global averages.
    """

    print("Building customer behavior profiles...")

    # Sort by time first
    transactions_df = transactions_df.sort_values('timestamp')

    profiles = []

    for customer_id, group in transactions_df.groupby('customer_id'):

        # Only use legitimate transactions to build profile (label 0 and 2)
        # We don't want fraud transactions polluting the profile
        legit = group[group['label'] != 1]

        if len(legit) == 0:
            continue

        profile = {
            'customer_id': customer_id,

            # Amount behavior
            'profile_avg_amount': legit['amount'].mean(),
            'profile_std_amount': legit['amount'].std() if len(legit) > 1 else 0,
            'profile_max_amount': legit['amount'].max(),
            'profile_min_amount': legit['amount'].min(),

            # Time behavior
            'profile_avg_hour': legit['hour'].mean(),
            'profile_std_hour': legit['hour'].std() if len(legit) > 1 else 0,

            # Day behavior
            'profile_avg_day_of_week': legit['day_of_week'].mean(),

            # Location behavior
            'profile_foreign_rate': legit['is_foreign_country'].mean(),

            # Merchant behavior
            'profile_new_merchant_rate': legit['is_new_merchant'].mean(),
            'profile_favorite_category': legit['merchant_category'].mode()[0],

            # Channel behavior
            'profile_online_rate': (legit['channel'] == 'online').mean(),

            # Transaction frequency
            'profile_total_transactions': len(legit),
            'profile_avg_daily_transactions': len(legit) / 365,
        }

        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)
    print(f"Built profiles for {len(profiles_df)} customers")
    return profiles_df


def merge_profiles_with_transactions(transactions_df, profiles_df):
    """
    Join each transaction with its customer's profile.
    Now every transaction row has both the transaction details
    AND the customer's historical behavior — ready for feature engineering.
    """
    merged = transactions_df.merge(profiles_df, on='customer_id', how='left')
    print(f"Merged transactions with profiles: {len(merged)} rows")
    return merged


if __name__ == "__main__":
    # Test it
    transactions_df = pd.read_csv('data/transactions.csv')
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])

    profiles_df = build_customer_profiles(transactions_df)
    profiles_df.to_csv('data/customer_profiles.csv', index=False)

    merged_df = merge_profiles_with_transactions(transactions_df, profiles_df)
    merged_df.to_csv('data/merged_transactions.csv', index=False)

    print("\nSample profile:")
    print(profiles_df.head(3).to_string())
    print("\nDone. Saved to data/customer_profiles.csv and data/merged_transactions.csv")