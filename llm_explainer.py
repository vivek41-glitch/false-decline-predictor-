import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def generate_llm_explanation(transaction, fraud_prob, fd_prob, shap_reasons, decision):
    """
    Takes ML model output and generates a human-readable
    paragraph explanation using LLaMA via Groq.
    This is the key differentiator — turns numbers into actionable insights.
    """

    shap_text = "\n".join(shap_reasons) if shap_reasons else "No SHAP data available"

    prompt = f"""You are an expert fraud analyst at a financial institution.
A transaction has been flagged by our AI system. Analyze it and write a 
clear, professional 3-4 sentence explanation for a fraud analyst.

Transaction Details:
- Amount: ${transaction.get('amount', transaction.get('Amount', 0)):.2f}
- Hour of day: {transaction.get('hour', 0)}:00
- Is high amount: {transaction.get('is_high_amount', 0)}
- Is late night: {transaction.get('is_late_night', 0)}
- Is high velocity: {transaction.get('is_high_velocity', 0)}

Model Scores:
- Fraud probability: {fraud_prob:.1%}
- False decline probability: {fd_prob:.1%}
- System decision: {decision}

Top contributing factors (SHAP):
{shap_text}

Write a professional explanation covering:
1. Why this transaction was flagged or approved
2. What specific signals drove the decision
3. Recommended action for the analyst
Keep it concise, professional, and actionable. No bullet points — write as paragraph."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )

    return response.choices[0].message.content


def generate_batch_summary(results):
    """
    Summarize a batch of transactions in plain English.
    Shows the big picture to management.
    """
    total = len(results)
    approved = sum(1 for r in results if r['decision'] == 'AUTO APPROVE')
    declined = sum(1 for r in results if r['decision'] == 'AUTO DECLINE')
    review = sum(1 for r in results if r['decision'] == 'HUMAN REVIEW')
    avg_fraud = sum(r['fraud_prob'] for r in results) / total if total > 0 else 0
    avg_fd = sum(r['fd_prob'] for r in results) / total if total > 0 else 0

    prompt = f"""You are a fraud analyst writing a brief management summary.
    
Batch of {total} transactions just processed:
- Auto approved: {approved} ({approved/total*100:.1f}%)
- Auto declined: {declined} ({declined/total*100:.1f}%)
- Sent to human review: {review} ({review/total*100:.1f}%)
- Average fraud risk score: {avg_fraud:.3f}
- Average false decline risk: {avg_fd:.3f}

Write a 2-3 sentence executive summary of this batch.
Highlight any concerns or notable patterns. Keep it professional."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.3
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("Testing LLM explanation layer...")

    test_txn = {
        'amount': 4500.00,
        'hour': 3,
        'is_high_amount': 1,
        'is_late_night': 1,
        'is_high_velocity': 1
    }

    explanation = generate_llm_explanation(
        transaction=test_txn,
        fraud_prob=0.87,
        fd_prob=0.12,
        shap_reasons=[
            'Is Late Night increased risk (impact: 0.342)',
            'Is High Amount increased risk (impact: 0.287)',
            'Is High Velocity increased risk (impact: 0.198)'
        ],
        decision='AUTO DECLINE'
    )

    print("\n--- LLM Explanation ---")
    print(explanation)

    print("\n--- Batch Summary ---")
    test_results = [
        {'decision': 'AUTO APPROVE', 'fraud_prob': 0.02, 'fd_prob': 0.01},
        {'decision': 'AUTO APPROVE', 'fraud_prob': 0.05, 'fd_prob': 0.03},
        {'decision': 'AUTO DECLINE', 'fraud_prob': 0.92, 'fd_prob': 0.08},
        {'decision': 'HUMAN REVIEW', 'fraud_prob': 0.45, 'fd_prob': 0.40},
    ]
    summary = generate_batch_summary(test_results)
    print(summary)
    print("\n✅ LLM explainer working!")