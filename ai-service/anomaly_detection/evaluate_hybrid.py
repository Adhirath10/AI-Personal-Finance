"""Evaluate the anomaly detector exactly as the app uses it.

The app does not show raw Isolation Forest output: it combines the behavioral
Isolation Forest with checks against the user's own history (large for the
category, possible duplicate charge, large share of income, new merchant) and
gives each flag a severity. This script scores every held-out expense (after
the model's training_end) with that combined detector and compares four
variants with the stored labels:

* Isolation Forest only (the model on its own)
* History checks only (no model)
* Combined, every flag shown in the app (a model-only flag needs a score
  that is unusual for this user: above their own earlier 98th percentile)
* Combined, Medium and High severity only

Stored labels have no verified provenance, so these numbers measure agreement
with the stored annotations, not fraud-detection performance.

Run from the project root:

    python ai-service/anomaly_detection/evaluate_hybrid.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
# Appended, not prepended: recommendation/ has its own evaluate.py, which must
# not shadow anomaly_detection/evaluate.py.
sys.path.append(str(AI_SERVICE_DIR / "recommendation"))

from engine import (  # noqa: E402
    DEFAULT_ANOMALY_MODEL_DIR,
    DEFAULT_DATABASE_PATH,
    RecommendationEngine,
    load_user_financial_data,
)

VARIANTS = {
    "Isolation Forest only": None,  # raw model flags, see evaluate()
    "History checks only": lambda flag: flag["detected_by"] in {"rules", "model + rules"},
    "Combined (all flags shown in the app)": lambda flag: True,
    "Combined, Medium and High severity only": lambda flag: flag["severity"] in {"Medium", "High"},
}


def _labels(database_path: Path, training_end: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{database_path.resolve().as_posix()}?mode=ro", uri=True) as connection:
        columns = {row[1] for row in connection.execute('PRAGMA table_info("transactions")')}
        if "is_anomaly" not in columns:
            raise ValueError("The transactions table has no is_anomaly labels to compare with.")
        anomaly_type = "anomaly_type" if "anomaly_type" in columns else "NULL AS anomaly_type"
        return pd.read_sql_query(
            f"""
            SELECT transaction_id, user_id, is_anomaly, {anomaly_type}
            FROM transactions
            WHERE lower(trim(transaction_type)) = 'expense' AND date(transaction_date) > date(?)
            """,
            connection,
            params=(training_end,),
        )


def _metrics(predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
    true_positive = int((predicted & actual).sum())
    flagged = int(predicted.sum())
    positives = int(actual.sum())
    precision = true_positive / flagged if flagged else 0.0
    recall = true_positive / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": flagged / len(predicted) if len(predicted) else 0.0,
        "flagged": flagged,
        "caught": true_positive,
    }


def evaluate(database_path: str | Path | None = None, model_dir: str | Path | None = None) -> dict:
    database = Path(database_path or DEFAULT_DATABASE_PATH)
    engine = RecommendationEngine(database_path=database, anomaly_model_dir=model_dir or DEFAULT_ANOMALY_MODEL_DIR)
    artifacts = engine._load_anomaly_artifacts()
    if artifacts is None:
        raise FileNotFoundError(f"Anomaly model artifacts are missing from {engine.anomaly_model_dir}.")
    training_end = str(artifacts["metadata"]["training_end"])[:10]

    labels = _labels(database, training_end)
    if labels.empty:
        raise ValueError(f"No expense transactions after the model's training end ({training_end}).")
    labels["is_anomaly"] = pd.to_numeric(labels["is_anomaly"], errors="coerce").fillna(0).astype(int).eq(1)

    flags: dict[str, dict] = {}
    model_flagged: set[str] = set()
    for user_id in sorted(labels["user_id"].astype(str).unique()):
        transactions = load_user_financial_data(database, user_id)["transactions"].copy()
        transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], errors="coerce")
        transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce")
        transactions = transactions.dropna(subset=["transaction_date", "amount"])
        result = engine._score_potential_anomalies(transactions)
        model_flagged.update(result.get("model_flagged_ids", []))
        for item in result.get("transactions", []):
            flags[str(item["transaction_id"])] = item

    actual = labels["is_anomaly"]
    results = {}
    for name, rule in VARIANTS.items():
        if rule is None:
            predicted = labels["transaction_id"].astype(str).isin(model_flagged)
        else:
            predicted = labels["transaction_id"].astype(str).map(lambda tid: tid in flags and bool(rule(flags[tid])))
        overall = _metrics(predicted, actual)
        by_type = {}
        for anomaly_type, group in labels.loc[actual].groupby(labels["anomaly_type"].fillna("unspecified")):
            caught = predicted.loc[group.index]
            by_type[str(anomaly_type)] = {"recall": float(caught.mean()), "count": int(len(group))}
        results[name] = {**overall, "recall_by_type": by_type}

    # Which history check raises how many alerts, and how many of them match a label.
    labelled = set(labels.loc[actual, "transaction_id"].astype(str))
    signals: dict[str, dict[str, int]] = {}
    for tid, flag in flags.items():
        for signal in flag.get("signals", []):
            entry = signals.setdefault(signal, {"flags": 0, "labelled": 0, "alone": 0})
            entry["flags"] += 1
            entry["labelled"] += tid in labelled
            entry["alone"] += set(flag["signals"]) <= {signal, "model"}
    return {
        "signals": signals,
        "training_end": training_end,
        "evaluated_rows": int(len(labels)),
        "labelled_anomalies": int(actual.sum()),
        "results": results,
    }


def format_report(summary: dict) -> str:
    lines = [
        "Combined Anomaly Detector Evaluation (as used by the app)",
        "=" * 56,
        "Stored-label reliability: UNVERIFIED. Metrics measure agreement with",
        "stored annotations only; they are not fraud-detection performance.",
        "",
        f"Held-out expenses (after training end {summary['training_end']}): {summary['evaluated_rows']:,}",
        f"Stored anomaly labels among them: {summary['labelled_anomalies']:,}",
        "",
        f"{'Detector':42} {'Precision':>9} {'Recall':>7} {'F1':>6} {'Alerts':>7}",
    ]
    for name, values in summary["results"].items():
        lines.append(
            f"{name:42} {values['precision']:9.3f} {values['recall']:7.3f} {values['f1']:6.3f} {values['alert_rate']:7.2%}"
        )
    lines += ["", "Alerts raised by each check (a flag can have several checks):",
              f"  {'Check':24} {'Flags':>6} {'Match a label':>14} {'Only check on the flag':>23}"]
    for signal, item in sorted(summary.get("signals", {}).items(), key=lambda pair: -pair[1]["flags"]):
        share = item["labelled"] / item["flags"] if item["flags"] else 0
        lines.append(f"  {signal:24} {item['flags']:6,} {item['labelled']:6,} ({share:5.1%}) {item['alone']:15,}")
    lines += ["", "Recall by stored anomaly type:"]
    for name, values in summary["results"].items():
        lines.append(f"  {name}:")
        for anomaly_type, item in sorted(values["recall_by_type"].items()):
            lines.append(f"    {anomaly_type}: {item['recall']:.1%} of {item['count']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database-path", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output", default=None, help="Report path (default: hybrid_evaluation_report.txt in the model folder).")
    args = parser.parse_args()
    try:
        summary = evaluate(args.database_path, args.model_dir)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    report = format_report(summary)
    output = Path(args.output) if args.output else Path(args.model_dir or DEFAULT_ANOMALY_MODEL_DIR) / "hybrid_evaluation_report.txt"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
