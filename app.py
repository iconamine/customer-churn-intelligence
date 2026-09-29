"""Interactive decision interface for a hypothetical customer scenario."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from churn_intelligence.data import generate_customer_data
from churn_intelligence.model import FEATURES, train_and_evaluate

st.set_page_config(page_title="Churn Intelligence", page_icon="📊", layout="wide")


@st.cache_resource
def trained_model():
    data = generate_customer_data(5_000, 42)
    return train_and_evaluate(data, 42)


result = trained_model()
st.title("Customer Churn Intelligence")
st.caption("Explainable risk scoring on synthetic portfolio data — no confidential records.")

metric_cols = st.columns(4)
for column, (name, value) in zip(metric_cols, result.metrics.items()):
    if name != "observed_churn_rate":
        column.metric(name.replace("_", " ").title(), f"{value:.1%}")

st.subheader("Score a customer scenario")
left, right = st.columns(2)
with left:
    tenure = st.slider("Tenure (months)", 1, 72, 18)
    charges = st.slider("Monthly charges", 18, 145, 75)
    tickets = st.slider("Support tickets — last 6 months", 0, 10, 2)
    late = st.slider("Late payments — last 12 months", 0, 8, 1)
with right:
    contract = st.selectbox("Contract", ["month-to-month", "one-year", "two-year"])
    internet = st.selectbox("Internet service", ["fiber", "dsl", "none"])
    payment = st.selectbox("Payment method", ["electronic-check", "bank-transfer", "credit-card"])
    dependents = st.selectbox("Has dependents", ["no", "yes"])

customer = pd.DataFrame(
    [
        {
            "tenure_months": tenure,
            "monthly_charges": charges,
            "support_tickets_6m": tickets,
            "late_payments_12m": late,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "has_dependents": dependents,
        }
    ]
)
risk = result.pipeline.predict_proba(customer[FEATURES])[:, 1][0]
st.metric("Estimated churn risk", f"{risk:.1%}")
st.progress(float(risk))
if risk >= 0.65:
    st.error("High priority — investigate service friction and propose a retention action.")
elif risk >= 0.35:
    st.warning("Medium priority — monitor and consider proactive outreach.")
else:
    st.success("Low priority — continue monitoring customer health.")

st.subheader("Highest-risk customers in the evaluation sample")
st.dataframe(result.retention_queue.head(20), use_container_width=True, hide_index=True)

