import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import os

from feature_engineering import engineer_features
from customer_profiler import build_customer_profiles, merge_profiles_with_transactions


def generate_shap_explanations(model, X, feature_cols, model_name="model", max_samples=500):
    """
    SHAP tells us WHY the model made each decision.
    Instead of a black box, every decision has a human-readable explanation.
    This is critical for financial systems — regulators require explainability.
    """
    print(f"\nGenerating SHAP explanations for {model_name}...")

    # Use a sample for speed
    X_sample = X.sample(min(max_samples, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle both binary and multiclass output
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 = positive class
    else:
        sv = shap_values

    os.makedirs('data/shap_plots', exist_ok=True)

    # ── PLOT 1: Feature importance (bar chart) ────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, feature_names=feature_cols,
                      plot_type="bar", show=False)
    plt.title(f"{model_name} — Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f'data/shap_plots/{model_name}_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: data/shap_plots/{model_name}_importance.png")

    # ── PLOT 2: Summary dot plot ──────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_cols, show=False)
    plt.title(f"{model_name} — SHAP Summary")
    plt.tight_layout()
    plt.savefig(f'data/shap_plots/{model_name}_summary.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: data/shap_plots/{model_name}_summary.png")

    return explainer, shap_values, X_sample


def explain_single_transaction(model, explainer, transaction_row, feature_cols, model_name="model"):
    """
    Explain ONE specific transaction in plain english.
    Why was this flagged? Why was it approved?
    This is what the dashboard will show for each transaction.
    """
    x = transaction_row[feature_cols].fillna(0).values.reshape(1, -1)
    x_df = pd.DataFrame(x, columns=feature_cols)

    shap_values = explainer.shap_values(x_df)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Get top 5 factors driving the decision
    feature_impacts = list(zip(feature_cols, sv))
    feature_impacts.sort(key=lambda x: float(np.mean(np.abs(x[1]))), reverse=True)
    top_factors = feature_impacts[:5]

    # Build human readable explanation
    explanations = []
    for feat, impact in top_factors:
        direction = "increased" if float(np.mean(impact)) > 0 else "decreased"
    explanations.append(f"• {feat.replace('_', ' ').title()} {direction} risk (impact: {float(np.mean(impact)):.3f})")
    return explanations, top_factors


if __name__ == "__main__":
    print("Loading data and models...")
    transactions_df = pd.read_csv('data/transactions.csv')
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])

    profiles_df = build_customer_profiles(transactions_df)
    merged_df = merge_profiles_with_transactions(transactions_df, profiles_df)
    df_featured, feature_cols = engineer_features(merged_df)

    X = df_featured[feature_cols].fillna(0)

    # Load models
    fraud_model = joblib.load('models/fraud_model.pkl')
    false_decline_model = joblib.load('models/false_decline_model.pkl')

    # Generate SHAP plots for both models
    fraud_explainer, _, _ = generate_shap_explanations(
        fraud_model, X, feature_cols, "fraud_model"
    )
    fd_explainer, _, _ = generate_shap_explanations(
        false_decline_model, X, feature_cols, "false_decline_model"
    )

    # Save explainers
    joblib.dump(fraud_explainer, 'models/fraud_explainer.pkl')
    joblib.dump(fd_explainer, 'models/fd_explainer.pkl')

    # Test single transaction explanation
    print("\n--- Sample Transaction Explanation ---")
    sample_txn = df_featured[df_featured['label'] == 2].iloc[0]
    explanations, _ = explain_single_transaction(
        false_decline_model, fd_explainer, sample_txn, feature_cols, "false_decline_model"
    )
    print("Why this transaction was flagged as possible false decline:")
    for e in explanations:
        print(e)

    print("\n✅ SHAP explanations done. Plots saved to data/shap_plots/")