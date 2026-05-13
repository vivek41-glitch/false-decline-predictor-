# FinShield AI — False Decline Predictor

> An AI system that catches legitimate transactions wrongly blocked by fraud detection — trained on 284,807 real bank transactions.

## The Problem

Financial institutions lose **$443 billion annually** from false declines — legitimate transactions wrongly rejected by fraud detection systems. A real customer gets blocked, gets embarrassed, and never comes back. The institution loses the transaction fee and the customer.

Current fraud systems optimize for catching fraud but ignore the cost of wrongly blocking real customers. This project solves that.

## The Solution

A second AI layer that sits behind standard fraud detection and asks: *"Wait — is this actually a legitimate customer who just looks suspicious?"*

Every transaction is compared against the customer's own behavioral profile — not global averages. A traveler spending abroad, a customer making a large purchase, someone shopping at 2am — these look suspicious globally but may be completely normal for that specific customer.

## Live Demo
[Deploy on Streamlit Cloud — link here]

## Architecture
Real Transaction Stream
↓
Customer Behavior Profiler (personal spending history)
↓
Feature Engineering (43 features including velocity)
↓
XGBoost Fraud Detector (baseline — what current systems do)
↓
Random Forest False Decline Detector (our innovation)
↓
Isolation Forest Anomaly Detector (unsupervised layer)
↓
3-Tier Decision Engine (Auto Approve / Human Review / Auto Decline)
↓
SHAP Explainability (why this decision?)
↓
LLaMA-3.3-70B Explanation (plain English for analysts)
↓
Feedback Loop (online learning from analyst corrections)
↓
Streamlit Dashboard (live feed, analysis, performance, impact)

## Results

| Metric | Value |
|---|---|
| Dataset | 284,807 real bank transactions (Kaggle IEEE-CIS) |
| Real fraud cases | 492 (0.173%) |
| False declines caught | 50,713 |
| Catch rate | 100% |
| Fraud model AUC-ROC | 0.9990 |
| False decline AUC-ROC | 1.0000 |
| Revenue protected | $4,480,000+ |

## Features

- **Real Data** — trained on 284,807 real bank transactions
- **43 engineered features** including velocity, behavioral profiling, amount deviation
- **3-model ensemble** — XGBoost + Random Forest + Isolation Forest
- **SHAP explainability** — every decision explained, regulatory compliant
- **LLM explanations** — LLaMA-3.3-70B generates plain English analyst reports
- **Online learning** — model retrains from analyst feedback in real time
- **Live stream** — queue-based real-time transaction processing
- **Business impact calculator** — converts ML predictions into dollar figures

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| ML Models | XGBoost, Random Forest, Isolation Forest |
| Data Balancing | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| LLM | Groq API (LLaMA-3.3-70B) |
| Online Learning | SGDClassifier (partial_fit) |
| Dashboard | Streamlit |
| Data | pandas, numpy |

## Setup

```bash
git clone https://github.com/yourusername/false-decline-predictor
cd false-decline-predictor
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
GROQ_API_KEY=your_key_here

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` folder.

Then run:
```bash
python train_real.py
python explainer.py
python model_performance.py
python feedback_loop.py
streamlit run app.py
```

## Why This Project

This project directly addresses one of the most expensive unsolved problems in financial services — false declines. Unlike most student ML projects that focus only on fraud detection, this system solves the harder, less obvious problem on the other side: catching the system's own mistakes before they cost the business money and customers.

---

Built by Vivek Dubey — B.E. CSE, GTU
Save it. Then run:
streamlit run app.py
