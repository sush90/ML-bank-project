# ML-bank-project
# 🦬 Bucknell Lending Club — AI Loan Evaluator

> *What if a bank could predict the future?*

---

## The Problem

Every year, lending institutions lose billions of dollars to loan defaults. The challenge isn't just identifying *bad* borrowers — it's finding the ones who *look* good on paper but carry hidden risk, and more importantly, not turning away the ones who will pay back every penny.

We built a machine learning system that does exactly that.

---

## What We Built

Using a real-world dataset of thousands of Lending Club loan applications, we trained two models that work together to evaluate any loan application in real time:

- **A Logistic Regression Classifier** - predicts the probability that a borrower will fully repay their loan
- **A LASSO Regressor** - predicts the pessimistic annualized return if the loan is funded

Feed it a borrower's financial profile. It tells you whether to approve, review, or decline — instantly.

---

## How It Works

```
Borrower fills out application
        ↓
Model evaluates 28+ financial features
        ↓
Classifier → "71% chance this loan gets fully repaid"
Regressor  → "Expected pessimistic return: +3.2%"
        ↓
Recommendation: ✅ APPROVE
```

The models don't just guess - they were trained, validated, and tested on historical Lending Club data using industry-standard techniques including polynomial feature expansion, log transformations, and regularization to prevent overfitting.

---

## The Stack

| Layer | Technology |
|---|---|
| Model Training | Python, scikit-learn |
| Web App | Streamlit |
| REST API | FastAPI + Uvicorn |
| Deployment | Render / Streamlit Cloud |
| Data | Lending Club Historical + Current Dataset |

---

## Files in This Repo

```
📁 bucknell-lending-club/
├── app.py                  ← Streamlit web application
├── main.py                 ← FastAPI REST endpoint
├── classifier.pkl          ← Trained logistic regression model
├── scaler_clf.pkl          ← Feature scaler for classifier
├── regressor.pkl           ← Trained LASSO regression model
├── scaler_reg.pkl          ← Feature scaler for regressor
├── requirements.txt        ← Python dependencies
└── README.md               ← You are here
```

---

## Live Demo

🌐 **Streamlit App:**http://localhost:8503/

---

## Try the API ### Under...Construction....

Once deployed, you can hit the `/predict` endpoint with any loan application:

```python
import requests

application = {
    "loan_amnt": 10000,
    "term_num": 36,
    "int_rate": 12.0,
    "grade": "B",
    "emp_length": "10+ years",
    "home_ownership": "RENT",
    "annual_inc": 65000,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 15.0,
    "delinq_2yrs": 0,
    "open_acc": 10,
    "pub_rec": 0,
    "fico_range_high": 704,
    "fico_range_low": 700,
    "revol_bal": 8000,
    "revol_util": 40.0
}

response = requests.post("https://your-render-url.onrender.com/predict", json=application)
print(response.json())
```

```json
{
  "prob_fully_paid": 0.823,
  "prob_default": 0.177,
  "predicted_return": 3.24,
  "recommendation": "APPROVE"
}
```

---

## The Recommendation Engine

| Condition | Decision |
|---|---|
| Repay probability ≥ 75% **and** return ≥ 0% | ✅ Approve |
| Repay probability ≥ 55% **and** return ≥ -2% | ⚠️ Manual Review |
| Anything below | ❌ Decline |

---

## Built By

**Sushma Upadhayay** & **Richard Perez**
Bucknell University · ANOP 330 · Spring 2026

*This project was built for educational purposes as part of a business analytics final project. Not financial advice.*
