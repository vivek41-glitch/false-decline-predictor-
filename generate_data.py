import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

# ─── CONFIG ───────────────────────────────────────────────
NUM_CUSTOMERS = 500
NUM_TRANSACTIONS = 50000
FRAUD_RATE = 0.02        # 2% fraud
FALSE_DECLINE_RATE = 0.05  # 5% legitimate but looks suspicious

MERCHANT_CATEGORIES = [
    'grocery', 'restaurant', 'travel', 'entertainment',
    'shopping', 'healthcare', 'fuel', 'electronics', 'hotel', 'online'
]

COUNTRIES = ['India', 'India', 'India', 'India', 'USA', 'UAE', 'UK', 'Singapore']

# ─── STEP 1: CREATE CUSTOMER PROFILES ─────────────────────
def create_customers():
    customers = []
    for i in range(NUM_CUSTOMERS):
        customers.append({
            'customer_id': f'CUST_{i:04d}',
            'name': fake.name(),
            'home_country': 'India',
            'avg_transaction_amount': random.uniform(200, 5000),
            'typical_hour_start': random.randint(8, 11),
            'typical_hour_end': random.randint(18, 22),
            'favorite_category': random.choice(MERCHANT_CATEGORIES),
            'monthly_transaction_count': random.randint(5, 40),
        })
    return pd.DataFrame(customers)

# ─── STEP 2: GENERATE TRANSACTIONS ────────────────────────
def generate_transactions(customers_df):
    transactions = []
    start_date = datetime(2024, 1, 1)

    for _ in range(NUM_TRANSACTIONS):
        customer = customers_df.sample(1).iloc[0]
        rand = random.random()

        # Decide transaction type
        is_fraud = rand < FRAUD_RATE
        is_false_decline_candidate = (not is_fraud) and (rand < FRAUD_RATE + FALSE_DECLINE_RATE)

        # Transaction time
        if is_fraud:
            # Fraudsters prefer late night
            hour = random.choice([0, 1, 2, 3, 4, 23])
        elif is_false_decline_candidate:
            # Real customer but unusual time (traveling etc)
            hour = random.choice([5, 6, 7, 22, 23])
        else:
            # Normal customer normal time
            hour = random.randint(
                customer['typical_hour_start'],
                customer['typical_hour_end']
            )

        # Transaction date
        days_offset = random.randint(0, 364)
        txn_date = start_date + timedelta(days=days_offset, hours=hour)

        # Amount
        if is_fraud:
            amount = random.uniform(5000, 50000)  # fraud = unusually high
        elif is_false_decline_candidate:
            amount = customer['avg_transaction_amount'] * random.uniform(2, 4)  # higher than usual but real
        else:
            amount = customer['avg_transaction_amount'] * random.uniform(0.5, 1.5)

        # Location
        if is_fraud:
            country = random.choice(['USA', 'UAE', 'UK', 'Singapore', 'Nigeria'])
        elif is_false_decline_candidate:
            country = random.choice(COUNTRIES)  # could be abroad = real traveler
        else:
            country = 'India'  # most normal transactions in home country

        # Merchant category
        if is_fraud:
            category = random.choice(['electronics', 'travel', 'online'])
        elif is_false_decline_candidate:
            category = random.choice(MERCHANT_CATEGORIES)
        else:
            # Prefer their favorite category
            category = random.choices(
                MERCHANT_CATEGORIES,
                weights=[3 if c == customer['favorite_category'] else 1 for c in MERCHANT_CATEGORIES]
            )[0]

        # Is new merchant
        is_new_merchant = 1 if (is_fraud or is_false_decline_candidate) and random.random() > 0.4 else 0

        # Online or in person
        channel = 'online' if (is_fraud and random.random() > 0.3) else random.choice(['online', 'in_person'])

        # Label
        # 0 = normal legitimate
        # 1 = fraud
        # 2 = false decline candidate (legitimate but suspicious looking)
        if is_fraud:
            label = 1
        elif is_false_decline_candidate:
            label = 2
        else:
            label = 0

        transactions.append({
            'transaction_id': fake.uuid4(),
            'customer_id': customer['customer_id'],
            'amount': round(amount, 2),
            'hour': hour,
            'day_of_week': txn_date.weekday(),  # 0=Monday, 6=Sunday
            'month': txn_date.month,
            'country': country,
            'is_foreign_country': 1 if country != 'India' else 0,
            'merchant_category': category,
            'is_new_merchant': is_new_merchant,
            'channel': channel,
            'avg_customer_amount': round(customer['avg_transaction_amount'], 2),
            'typical_hour_start': customer['typical_hour_start'],
            'typical_hour_end': customer['typical_hour_end'],
            'favorite_category': customer['favorite_category'],
            'timestamp': txn_date,
            'label': label
            # label 0 = legitimate normal
            # label 1 = fraud
            # label 2 = false decline candidate (legit but looks suspicious)
        })

    return pd.DataFrame(transactions)

# ─── STEP 3: RUN AND SAVE ─────────────────────────────────
if __name__ == "__main__":
    print("Creating customer profiles...")
    customers_df = create_customers()
    customers_df.to_csv('data/customers.csv', index=False)
    print(f"Created {len(customers_df)} customers")

    print("Generating transactions...")
    transactions_df = generate_transactions(customers_df)
    transactions_df.to_csv('data/transactions.csv', index=False)

    # Print summary
    print(f"\nTotal transactions: {len(transactions_df)}")
    print(f"Normal (label 0): {len(transactions_df[transactions_df.label==0])}")
    print(f"Fraud (label 1): {len(transactions_df[transactions_df.label==1])}")
    print(f"False decline candidates (label 2): {len(transactions_df[transactions_df.label==2])}")
    print(f"\nData saved to data/transactions.csv")
    print(f"Data saved to data/customers.csv")