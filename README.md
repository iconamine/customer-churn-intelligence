# Customer Churn Intelligence

An end-to-end, business-facing machine-learning project that turns customer behaviour into an explainable churn-risk score and an actionable retention queue.

> **Data note:** this public portfolio version uses reproducible synthetic data. It contains no customer, employer, or confidential information.

## Why this project matters

Churn models are useful only when they help a business decide **who to contact, why, and with what priority**. This repository therefore covers the full path from raw customer signals to a decision-ready output:

1. generate and validate a realistic customer dataset;
2. prepare numeric and categorical features in a leakage-safe pipeline;
3. train an interpretable logistic-regression baseline;
4. evaluate discrimination, precision, recall and calibration inputs;
5. rank customers by risk for retention action;
6. explore individual scenarios in a Streamlit dashboard.

## Architecture

```text
Synthetic customer signals
          │
          ▼
Data validation ──► Feature pipeline ──► Logistic regression
                                             │
                         ┌───────────────────┴──────────────────┐
                         ▼                                      ▼
                  Evaluation report                     Risk dashboard
```

## Tech stack

- Python, pandas, NumPy
- scikit-learn pipelines and model evaluation
- Streamlit for the decision interface
- pytest for automated tests
- Docker for reproducible execution
- GitHub Actions for continuous integration

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m churn_intelligence.cli --customers 5000 --output artifacts
streamlit run app.py
```

The training command writes:

- `artifacts/churn_model.joblib` — fitted preprocessing and model pipeline;
- `artifacts/metrics.json` — evaluation metrics;
- `artifacts/retention_queue.csv` — test customers ranked by churn risk.

## Quality controls

```bash
pytest
```

CI runs the test suite on every push and pull request. The project deliberately starts with an interpretable baseline before adding more complex models; this makes assumptions visible and gives stakeholders a defensible benchmark.

## Next iterations

- compare the baseline with gradient boosting using cross-validation;
- add probability calibration and drift monitoring;
- estimate intervention value using retention cost and customer lifetime value;
- expose batch scoring through a small REST API.

## Author

**Mohamed Amine Ajana** — Junior Data & IT Consultant based in Morocco.  
[LinkedIn](https://www.linkedin.com/in/amine-ajana-3a78a0376/) · [GitHub](https://github.com/iconamine)

