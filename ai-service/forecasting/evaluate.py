import argparse
import sqlite3
import sys
from pathlib import Path

from train import train_and_evaluate_model


def evaluate_model(database_path: str | Path | None = None, model_dir: str | Path | None = None):
    result = train_and_evaluate_model(database_path=database_path, model_dir=model_dir)
    print("Expense Forecasting Evaluation")
    print("=" * 32)
    print("Baseline (Moving Average)")
    print(f"MAE: {result['baseline_metrics']['mae']:.4f}")
    print(f"RMSE: {result['baseline_metrics']['rmse']:.4f}")
    print(f"MAPE: {result['baseline_metrics']['mape']:.4f}%")
    print()
    print("ML Model (Random Forest)")
    print(f"MAE: {result['ml_metrics']['mae']:.4f}")
    print(f"RMSE: {result['ml_metrics']['rmse']:.4f}")
    print(f"MAPE: {result['ml_metrics']['mape']:.4f}%")
    print()
    print(f"Better model: {'ML model' if result['ml_metrics']['mae'] <= result['baseline_metrics']['mae'] else 'Baseline'}")
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the monthly expense forecasting model using the real SQLite dataset.")
    parser.add_argument("--database-path", type=str, default=None, help="Path to the SQLite database file.")
    parser.add_argument("--model-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "models" / "forecasting"), help="Directory containing model artifacts.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        evaluate_model(args.database_path, args.model_dir)
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
