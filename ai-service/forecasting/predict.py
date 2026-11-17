import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from train import build_monthly_expense_data, create_forecast_dataset


def load_model(model_dir: str | Path | None = None):
    target_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parents[1] / "models" / "forecasting"
    if not (target_dir / "forecasting_model.pkl").exists():
        raise FileNotFoundError(f"Forecasting model not found in {target_dir}. Train the model first.")
    model = joblib.load(target_dir / "forecasting_model.pkl")
    feature_columns = joblib.load(target_dir / "feature_columns.pkl")
    return model, feature_columns


def predict_next_month_expense(user_id: str, database_path: str | Path | None = None, model_dir: str | Path | None = None):
    if not user_id or not str(user_id).strip():
        raise ValueError("Invalid user ID. Please provide a valid user ID.")

    monthly = build_monthly_expense_data(database_path)
    user_df = monthly[monthly["user_id"] == str(user_id)].copy()
    if user_df.empty:
        raise ValueError(f"User ID not found or no expense history: {user_id}")

    feature_dataset = create_forecast_dataset(monthly)
    last_row = feature_dataset[feature_dataset["user_id"] == str(user_id)].sort_values("month").tail(1)
    if last_row.empty:
        raise ValueError(f"User {user_id} does not have enough historical data to forecast the next month.")

    model, feature_columns = load_model(model_dir)
    features = last_row[feature_columns].iloc[[0]]
    predicted = float(model.predict(features)[0])
    forecast_month = pd.Period(last_row["month"].iloc[0], freq="M") + 1
    return str(user_id), forecast_month.strftime("%Y-%m"), predicted


def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict next month's total expense for a user.")
    parser.add_argument("--user-id", type=str, required=True, help="User ID to forecast.")
    parser.add_argument("--database-path", type=str, default=None, help="Path to the SQLite database.")
    parser.add_argument("--model-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "models" / "forecasting"), help="Directory containing trained model artifacts.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        user_id, forecast_month, predicted = predict_next_month_expense(args.user_id, args.database_path, args.model_dir)
        print(f"User ID: {user_id}")
        print(f"Forecast Month: {forecast_month}")
        print(f"Predicted Expense: ₹{predicted:,.2f}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
