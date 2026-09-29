"""Command-line training workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import generate_customer_data
from .model import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the churn model")
    parser.add_argument("--customers", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = generate_customer_data(args.customers, args.seed)
    result = train_and_evaluate(data, args.seed)

    joblib.dump(result.pipeline, args.output / "churn_model.joblib")
    (args.output / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    result.retention_queue.to_csv(args.output / "retention_queue.csv", index=False)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()

