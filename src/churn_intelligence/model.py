"""Leakage-safe preprocessing, training, evaluation, and prioritisation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import validate_customer_data

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_6m",
    "late_payments_12m",
]
CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
    "has_dependents",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    retention_queue: pd.DataFrame


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )


def train_and_evaluate(data: pd.DataFrame, seed: int = 42) -> TrainingResult:
    validate_customer_data(data)
    train, test = train_test_split(
        data,
        test_size=0.25,
        random_state=seed,
        stratify=data["churn"],
    )
    pipeline = build_pipeline()
    pipeline.fit(train[FEATURES], train["churn"])

    probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(test["churn"], probabilities)), 4),
        "accuracy": round(float(accuracy_score(test["churn"], predictions)), 4),
        "precision": round(float(precision_score(test["churn"], predictions)), 4),
        "recall": round(float(recall_score(test["churn"], predictions)), 4),
        "observed_churn_rate": round(float(test["churn"].mean()), 4),
    }

    queue = test[["customer_id", *FEATURES, "churn"]].copy()
    queue["churn_risk"] = probabilities.round(4)
    queue["priority"] = pd.cut(
        queue["churn_risk"],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["low", "medium", "high"],
    )
    queue = queue.sort_values("churn_risk", ascending=False).reset_index(drop=True)
    return TrainingResult(pipeline=pipeline, metrics=metrics, retention_queue=queue)

