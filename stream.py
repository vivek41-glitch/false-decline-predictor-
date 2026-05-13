import queue
import threading
import time
import pandas as pd
import numpy as np
import joblib
from real_data_loader import load_real_data, prepare_real_features, get_feature_columns

# Global transaction queue
transaction_queue = queue.Queue(maxsize=100)
results_queue = queue.Queue(maxsize=100)
is_streaming = False

def load_models():
    fraud_model = joblib.load('models/fraud_model.pkl')
    fd_model = joblib.load('models/false_decline_model.pkl')
    anomaly_model = joblib.load('models/anomaly_model.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return fraud_model, fd_model, anomaly_model, feature_cols

def get_decision(fraud_prob, fd_prob):
    if fraud_prob < 0.3 or fd_prob > 0.6:
        return 'AUTO APPROVE', 'green'
    elif fraud_prob > 0.7 and fd_prob < 0.3:
        return 'AUTO DECLINE', 'red'
    else:
        return 'HUMAN REVIEW', 'orange'

def transaction_producer(df, feature_cols, speed=0.3):
    """
    Produces transactions into the queue continuously.
    Simulates real payment network feed.
    """
    global is_streaming
    sample = df.sample(500, random_state=42).reset_index(drop=True)

    for idx, row in sample.iterrows():
        if not is_streaming:
            break
        transaction_queue.put(row)
        time.sleep(speed)

def transaction_processor(fraud_model, fd_model, anomaly_model, feature_cols):
    """
    Consumes transactions from queue, scores them, puts results in results queue.
    Runs in separate thread — non-blocking.
    """
    global is_streaming

    while is_streaming:
        try:
            txn = transaction_queue.get(timeout=1)
            x = txn[feature_cols].fillna(0).values.reshape(1, -1)

            fraud_prob = fraud_model.predict_proba(x)[0][1]
            fd_prob = fd_model.predict_proba(x)[0][1]
            anomaly = anomaly_model.predict(x)[0]
            decision, color = get_decision(fraud_prob, fd_prob)

            result = {
                'amount': round(float(txn.get('Amount', 0)), 2),
                'hour': int(txn.get('hour', 0)),
                'fraud_prob': round(float(fraud_prob), 4),
                'fd_prob': round(float(fd_prob), 4),
                'anomaly': int(anomaly),
                'decision': decision,
                'color': color,
                'true_label': int(txn.get('label', 0)),
                'timestamp': time.strftime('%H:%M:%S')
            }

            results_queue.put(result)
            transaction_queue.task_done()

        except queue.Empty:
            continue

def start_stream(df, feature_cols, speed=0.3):
    """Start the streaming pipeline in background threads."""
    global is_streaming
    is_streaming = True

    fraud_model, fd_model, anomaly_model, _ = load_models()

    producer = threading.Thread(
        target=transaction_producer,
        args=(df, feature_cols, speed),
        daemon=True
    )
    processor = threading.Thread(
        target=transaction_processor,
        args=(fraud_model, fd_model, anomaly_model, feature_cols),
        daemon=True
    )

    producer.start()
    processor.start()
    return producer, processor

def stop_stream():
    global is_streaming
    is_streaming = False

def get_results(max_results=20):
    results = []
    while not results_queue.empty() and len(results) < max_results:
        results.append(results_queue.get())
    return results

if __name__ == "__main__":
    print("Testing stream pipeline...")
    df = load_real_data()
    df = prepare_real_features(df)
    feature_cols = get_feature_columns()

    start_stream(df, feature_cols, speed=0.1)
    print("Stream started. Processing transactions...")
    time.sleep(3)

    results = get_results()
    print(f"\nProcessed {len(results)} transactions in 3 seconds:")
    for r in results[:5]:
        print(f"  ${r['amount']:>8.2f} | Hour:{r['hour']:>2} | "
              f"Fraud:{r['fraud_prob']:.3f} | FD:{r['fd_prob']:.3f} | "
              f"{r['decision']}")

    stop_stream()
    print("\n✅ Stream test complete!")
    