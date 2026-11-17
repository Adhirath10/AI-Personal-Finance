from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Explain existing personal-finance model predictions with SHAP."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    forecast_user = commands.add_parser("forecast-user")
    forecast_user.add_argument("--user-id", required=True)
    forecast_user.add_argument("--database-path")
    forecast_user.add_argument("--model-dir")
    forecast_user.add_argument("--output-dir")

    forecast_global = commands.add_parser("forecast-global")
    forecast_global.add_argument("--database-path")
    forecast_global.add_argument("--model-dir")
    forecast_global.add_argument("--output-dir")
    forecast_global.add_argument("--max-rows", type=int, default=500)

    category = commands.add_parser("categorize")
    category.add_argument("--description", default="")
    category.add_argument("--merchant", default="")
    category.add_argument("--class-label")
    category.add_argument("--database-path")
    category.add_argument("--model-dir")
    category.add_argument("--output-dir")

    category_global = commands.add_parser("categorize-global")
    category_global.add_argument("--database-path")
    category_global.add_argument("--model-dir")
    category_global.add_argument("--output-dir")
    category_global.add_argument("--max-rows", type=int, default=200)

    anomaly = commands.add_parser("anomaly-transaction")
    anomaly.add_argument("--transaction-id", required=True)
    anomaly.add_argument("--database-path")
    anomaly.add_argument("--model-dir")
    anomaly.add_argument("--output-dir")

    anomaly_global = commands.add_parser("anomaly-global")
    anomaly_global.add_argument(
        "--scope", choices=("personal", "type", "pooled"), default="pooled"
    )
    anomaly_global.add_argument("--user-id")
    anomaly_global.add_argument("--transaction-type", default="Expense")
    anomaly_global.add_argument("--database-path")
    anomaly_global.add_argument("--model-dir")
    anomaly_global.add_argument("--output-dir")
    anomaly_global.add_argument("--max-rows", type=int, default=100)

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        if args.command.startswith("forecast-"):
            from explainability.forecasting import (
                explain_forecast_for_user,
                explain_forecast_global,
            )

            if args.command == "forecast-user":
                result = explain_forecast_for_user(
                    args.user_id,
                    database_path=args.database_path,
                    model_dir=args.model_dir,
                    output_dir=args.output_dir,
                )
            else:
                result = explain_forecast_global(
                    database_path=args.database_path,
                    model_dir=args.model_dir,
                    output_dir=args.output_dir,
                    max_rows=args.max_rows,
                )
        elif args.command.startswith("categorize"):
            from explainability.categorization import (
                explain_category,
                explain_categorization_global,
            )

            if args.command == "categorize":
                result = explain_category(
                    args.description,
                    args.merchant,
                    database_path=args.database_path,
                    model_dir=args.model_dir,
                    output_dir=args.output_dir,
                    class_label=args.class_label,
                )
            else:
                result = explain_categorization_global(
                    database_path=args.database_path,
                    model_dir=args.model_dir,
                    output_dir=args.output_dir,
                    max_rows=args.max_rows,
                )
        else:
            from explainability.anomaly_detection import (
                explain_anomaly_global,
                explain_anomaly_transaction,
            )

            if args.command == "anomaly-transaction":
                result = explain_anomaly_transaction(
                    args.transaction_id,
                    database_path=args.database_path,
                    model_dir=args.model_dir,
                    output_dir=args.output_dir,
                )
            else:
                result = explain_anomaly_global(
                    database_path=args.database_path,
                    model_dir=args.model_dir,
                    output_dir=args.output_dir,
                    scope=args.scope,
                    user_id=args.user_id,
                    transaction_type=args.transaction_type,
                    max_rows=args.max_rows,
                )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        ValueError,
        RuntimeError,
        sqlite3.Error,
    ) as exc:
        print(f"Explanation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
