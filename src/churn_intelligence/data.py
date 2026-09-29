"""Reproducible synthetic customer data for the public portfolio project."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_customer_data(n_customers: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic, fully synthetic churn classification dataset."""
    if n_customers < 100:
        raise ValueError("n_customers must be at least 100")

    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n_customers)
    monthly_charges = np.clip(rng.normal(67, 24, n_customers), 18, 145).round(2)
    support_tickets = rng.poisson(1.5, n_customers)
    late_payments = rng.poisson(0.7, n_customers)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        n_customers,
        p=[0.56, 0.25, 0.19],
    )
    internet = rng.choice(["fiber", "dsl", "none"], n_customers, p=[0.50, 0.40, 0.10])
    payment = rng.choice(
        ["electronic-check", "bank-transfer", "credit-card"],
        n_customers,
        p=[0.38, 0.32, 0.30],
    )
    dependents = rng.choice(["yes", "no"], n_customers, p=[0.31, 0.69])

    risk = (
        -1.55
        - 0.026 * tenure
        + 0.015 * (monthly_charges - 65)
        + 0.20 * support_tickets
        + 0.34 * late_payments
        + 0.95 * (contract == "month-to-month")
        - 0.55 * (contract == "two-year")
        + 0.38 * (payment == "electronic-check")
        + 0.24 * (internet == "fiber")
        + 0.18 * (dependents == "no")
        + rng.normal(0, 0.45, n_customers)
    )
    churn_probability = 1 / (1 + np.exp(-risk))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(1, n_customers + 1)],
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "support_tickets_6m": support_tickets,
            "late_payments_12m": late_payments,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "has_dependents": dependents,
            "churn": churn,
        }
    )


def validate_customer_data(data: pd.DataFrame) -> None:
    """Fail fast when required columns, values, or identifiers are invalid."""
    required = {
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "support_tickets_6m",
        "late_payments_12m",
        "contract_type",
        "internet_service",
        "payment_method",
        "has_dependents",
        "churn",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data["customer_id"].duplicated().any():
        raise ValueError("customer_id must be unique")
    if not set(data["churn"].dropna().unique()).issubset({0, 1}):
        raise ValueError("churn must contain only 0 or 1")
    if data[list(required)].isna().any().any():
        raise ValueError("Required fields cannot contain missing values")

