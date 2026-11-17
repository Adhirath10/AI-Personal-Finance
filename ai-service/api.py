"""HTTP adapter for the existing, saved personal-finance AI artifacts."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_SERVICE_DIR = PROJECT_ROOT / "ai-service"
load_dotenv(PROJECT_ROOT / ".env")
DATABASE_PATH = Path(
    os.environ.get(
        "DATABASE_PATH",
        str(PROJECT_ROOT / "data" / "database" / "personal_finance.db"),
    )
).expanduser()
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = (PROJECT_ROOT / DATABASE_PATH).resolve()
MODEL_DIRS = {
    "categorization": AI_SERVICE_DIR / "models" / "categorization",
    "forecasting": AI_SERVICE_DIR / "models" / "forecasting",
    "anomaly": AI_SERVICE_DIR / "models" / "anomaly_detection_behavioral_experiment",
}
EXPLANATION_OUTPUT_DIR = Path(
    os.environ.get(
        "FINANCE_EXPLANATION_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "processed" / "explainability"),
    )
).expanduser()
if not EXPLANATION_OUTPUT_DIR.is_absolute():
    EXPLANATION_OUTPUT_DIR = (PROJECT_ROOT / EXPLANATION_OUTPUT_DIR).resolve()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finai.ai")
app = FastAPI(
    title="FinAI Model Service",
    description="Request-time adapters for the project's existing trained models.",
    version="1.0.0",
)


class CategorizeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[dict[str, str]] = Field(min_length=1, max_length=100)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load AI module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _recommendation_engine():
    module = _load_module(
        "_finai_recommendation_engine",
        AI_SERVICE_DIR / "recommendation" / "engine.py",
    )
    return module.RecommendationEngine(database_path=DATABASE_PATH)


def _analysis(
    user_id: str,
    *,
    include_forecast: bool = True,
    include_anomalies: bool = True,
) -> tuple[Any, dict[str, Any]]:
    engine = _recommendation_engine()
    return engine, engine.analyze_user(
        user_id,
        include_forecast=include_forecast,
        include_anomalies=include_anomalies,
    )


def _unavailable(exc: Exception, **extra: Any) -> dict[str, Any]:
    """Describe a failure in the response instead of returning a bare HTTP 500."""
    reason = str(exc) or type(exc).__name__
    return {"status": "unavailable", "reason": reason, **extra}


def _forecast_model_metrics() -> dict[str, float]:
    """Read the held-out MAE/RMSE/MAPE saved when the forecasting model was trained."""
    report_path = MODEL_DIRS["forecasting"] / "evaluation_report.txt"
    try:
        lines = report_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    metrics: dict[str, float] = {}
    in_model_section = False
    for line in lines:
        if line.startswith("Machine Learning Model"):
            in_model_section = True
            continue
        if not in_model_section:
            continue
        name, separator, value = line.partition(":")
        if not separator:
            break
        key = name.strip().casefold()
        if key in {"mae", "rmse", "mape"}:
            try:
                metrics[key] = float(value.strip().rstrip("%"))
            except ValueError:
                continue
    return metrics


def _forecast_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    """Expose the engine forecast, labelling forecasts for months that have passed.

    The project dataset is historical, so the month after the last recorded
    month is usually already in the past. The engine marks that as ``stale`` so
    it is not used for forward-looking recommendations, but the prediction is
    still a valid model output and is returned with an explicit notice.
    """
    result = dict(analysis["forecast"])
    result["analysis_period"] = analysis["analysis_period"]
    result["model"] = "RandomForestRegressor"
    status = result.get("status")
    if status in {"available", "stale"}:
        is_historical = status == "stale"
        history_end = analysis["analysis_period"]["selected_month"]
        result["status"] = "available"
        result["is_historical"] = is_historical
        result["history_end_month"] = history_end
        result["skip_reason"] = None
        result["notice"] = (
            f"Recorded expenses end in {history_end}, so this is the model's "
            f"forecast for {result.get('forecast_month')}, a month that has "
            "already passed. Add recent transactions to forecast an upcoming month."
            if is_historical
            else None
        )
        result.update(_forecast_model_metrics())
    else:
        result.setdefault("reason", result.get("skip_reason"))
    return result


def _explainability_module(model: str):
    module_path = AI_SERVICE_DIR / "explainability"
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))
    filenames = {
        "forecast": "forecasting.py",
        "forecasting": "forecasting.py",
        "categorization": "categorization.py",
        "anomaly": "anomaly_detection.py",
    }
    if model not in filenames:
        raise ValueError(f"Unsupported explanation model: {model}")
    return _load_module(
        f"_finai_explainability_{model}",
        module_path / filenames[model],
    )


def _recommendation_payload(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"recommendation-{index + 1}",
        "title": item["recommendation_title"],
        "description": item["recommendation_description"],
        "priority": item["priority"],
        "reason": item["reason"],
        "type": item["recommendation_type"],
        "supporting_values": item["supporting_values"],
    }


@app.get("/health")
def health() -> dict[str, str]:
    database_status = "unavailable"
    if DATABASE_PATH.is_file():
        try:
            with sqlite3.connect(
                f"file:{DATABASE_PATH.resolve().as_posix()}?mode=ro",
                uri=True,
            ) as connection:
                connection.execute("SELECT 1")
            database_status = "available"
        except sqlite3.Error:
            logger.exception("AI service database health check failed.")
    return {
        "status": "ok" if database_status == "available" else "unavailable",
        "database": database_status,
    }


@app.post("/api/categorize")
def categorize(body: CategorizeBody) -> dict[str, Any]:
    vectorizer_path = MODEL_DIRS["categorization"] / "tfidf_vectorizer.pkl"
    classifier_path = MODEL_DIRS["categorization"] / "category_classifier.pkl"
    if not vectorizer_path.is_file() or not classifier_path.is_file():
        return {"status": "unavailable", "reason": "Categorization model artifacts are missing.", "items": []}

    try:
        training = _load_module(
            "_finai_categorization_training",
            AI_SERVICE_DIR / "categorization" / "train.py",
        )
        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(classifier_path)
        results = []
        for index, item in enumerate(body.items):
            description = item.get("description", "").strip()
            merchant = item.get("merchant", "").strip()
            if not description and not merchant:
                raise HTTPException(
                    status_code=422,
                    detail=f"Item {index} must include a description or merchant.",
                )
            text = training.prepare_text_feature(description, merchant)
            features = vectorizer.transform([text])
            probabilities = classifier.predict_proba(features)[0]
            results.append(
                {
                    "id": item.get("id"),
                    "description": description,
                    "merchant": merchant,
                    "category": str(classifier.predict(features)[0]),
                    "confidence": float(probabilities.max()),
                }
            )
        return {"status": "available", "items": results}
    except HTTPException:
        raise
    except (OSError, EOFError, ValueError, ImportError) as exc:
        logger.exception("Categorization service could not use its saved artifacts.")
        return {"status": "unavailable", "reason": str(exc), "items": []}


@app.get("/api/forecast")
def forecast(user_id: str = Query(min_length=1, max_length=100)) -> dict[str, Any]:
    try:
        _, analysis = _analysis(user_id, include_anomalies=False)
        return _forecast_payload(analysis)
    except FileNotFoundError as exc:
        return _unavailable(exc)
    except Exception as exc:  # noqa: BLE001 - reported to the client
        logger.exception("Expense forecast could not be computed.")
        return _unavailable(exc)


@app.get("/api/anomalies")
def anomalies(user_id: str = Query(min_length=1, max_length=100)) -> dict[str, Any]:
    try:
        _, analysis = _analysis(user_id, include_forecast=False)
        result = analysis["potential_anomalies"]
        if result.get("status") == "unavailable":
            return {
                "status": "unavailable",
                "reason": result.get("skip_reason")
                or "The saved anomaly detection model is unavailable.",
                "analysis_period": analysis["analysis_period"],
                "items": [],
            }
        items = [
            {**item, "review_status": "Needs review"}
            for item in result.get("transactions", [])
        ]
        analyzed_count = int(result.get("analyzed_count", 0))
        return {
            "status": result.get("status", "unavailable"),
            "analysis_period": analysis["analysis_period"],
            "training_end": result.get("training_end"),
            "total_analyzed": analyzed_count,
            "normal_count": max(0, analyzed_count - len(items)),
            "anomaly_count": len(items),
            "alert_rate": len(items) / analyzed_count if analyzed_count else None,
            "items": items,
            "notice": "Unusual transactions are statistical outliers requiring review, not confirmed fraud.",
        }
    except FileNotFoundError as exc:
        return _unavailable(exc, items=[])
    except Exception as exc:  # noqa: BLE001 - reported to the client
        logger.exception("Anomaly analysis could not be computed.")
        return _unavailable(exc, items=[])


@app.get("/api/recommendations")
def recommendations(user_id: str = Query(min_length=1, max_length=100)) -> dict[str, Any]:
    try:
        engine, analysis = _analysis(user_id)
        items = [
            _recommendation_payload(item, index)
            for index, item in enumerate(engine.generate_recommendations(analysis))
        ]
        return {
            "status": "available",
            "analysis_period": analysis["analysis_period"],
            "items": items,
        }
    except FileNotFoundError as exc:
        return _unavailable(exc, items=[])
    except Exception as exc:  # noqa: BLE001 - reported to the client
        logger.exception("Recommendations could not be generated.")
        return _unavailable(exc, items=[])


@app.get("/api/insights")
def insights(user_id: str = Query(min_length=1, max_length=100)) -> dict[str, Any]:
    try:
        # Insights only compare recorded category spending, so they do not
        # depend on (or wait for) the forecasting and anomaly models.
        _, analysis = _analysis(
            user_id,
            include_forecast=False,
            include_anomalies=False,
        )
        items = []
        for row in analysis["category_trends"]:
            average = row.get("historical_average_category_spending")
            latest = row.get("latest_category_spending")
            if average is None or not average:
                continue
            change = (latest - average) / abs(average) * 100
            items.append(
                {
                    "id": f"category-{len(items) + 1}",
                    "title": f"{row['category']} spending changed",
                    "description": (
                        f"Recorded {row['category']} expenses for "
                        f"{analysis['analysis_month']} were "
                        f"{abs(change):.1f}% {'higher' if change >= 0 else 'lower'} "
                        "than the observed prior-month average."
                    ),
                    "priority": "Medium" if change > 20 else "Low",
                    "supporting_values": {
                        "category": row["category"],
                        "latest_month_spending": latest,
                        "historical_monthly_average": average,
                        "historical_months_compared": row["historical_months_compared"],
                    },
                }
            )
        items.sort(
            key=lambda item: abs(
                item["supporting_values"]["latest_month_spending"]
                - item["supporting_values"]["historical_monthly_average"]
            ),
            reverse=True,
        )
        period = analysis["analysis_period"]
        result = {
            "status": "available" if items else "no_insights",
            "analysis_period": period,
            "items": items,
            "notice": (
                f"Showing {period['selected_month']}, the latest month with recorded "
                f"expenses. Transactions in {period['latest_transaction_month']} "
                "contain no expenses yet."
                if period["selected_month"] != period["latest_transaction_month"]
                else None
            ),
        }
        if not items:
            result["reason"] = (
                f"No category in {analysis['analysis_month']} has earlier recorded "
                "spending to compare against yet."
            )
        return result
    except FileNotFoundError as exc:
        return _unavailable(exc, items=[])
    except Exception as exc:  # noqa: BLE001 - reported to the client
        logger.exception("Financial insights could not be computed.")
        return _unavailable(exc, items=[])


@app.get("/api/explanations")
def explanations(
    user_id: str = Query(min_length=1, max_length=100),
    model: str = Query(default="forecast", pattern="^(forecast|categorization|anomaly)$"),
    transaction_id: str | None = Query(default=None, max_length=100),
    description: str | None = Query(default=None, max_length=500),
    merchant: str | None = Query(default=None, max_length=200),
) -> dict[str, Any]:
    try:
        if model == "forecast":
            _, analysis = _analysis(user_id, include_anomalies=False)
            forecast_result = _forecast_payload(analysis)
            if forecast_result.get("status") != "available":
                return {
                    "status": forecast_result.get("status", "unavailable"),
                    "reason": forecast_result.get("reason"),
                    "analysis_period": analysis["analysis_period"],
                }
            module = _explainability_module("forecasting")
            output_dir = EXPLANATION_OUTPUT_DIR / "forecast"
            global_result = module.explain_forecast_global(
                database_path=DATABASE_PATH,
                model_dir=MODEL_DIRS["forecasting"],
                output_dir=output_dir,
                user_id=user_id,
            )
            individual = module.explain_forecast_for_user(
                user_id,
                database_path=DATABASE_PATH,
                model_dir=MODEL_DIRS["forecasting"],
                output_dir=output_dir,
            )
            return {
                "status": "available",
                "model": model,
                "forecast_month": individual["forecast_month"],
                "prediction": individual["prediction"],
                "prediction_value": individual["prediction"],
                "base_value": individual["base_value"],
                "analysis_period": analysis["analysis_period"],
                "is_historical": forecast_result["is_historical"],
                "notice": forecast_result.get("notice"),
                "feature_importance": [
                    {"feature": feature, "importance": value}
                    for feature, value in global_result["mean_absolute_shap"].items()
                ],
                "feature_contributions": individual["feature_contributions"],
                "explanation": individual["explanation"],
            }

        module = _explainability_module(model)
        output_dir = EXPLANATION_OUTPUT_DIR / model
        if model == "categorization":
            if not (description and description.strip()) and not (
                merchant and merchant.strip()
            ):
                if not transaction_id:
                    return {
                        "status": "unavailable",
                        "reason": "Provide a transaction ID or transaction text for a local categorization explanation.",
                    }
                with sqlite3.connect(
                    f"file:{DATABASE_PATH.resolve().as_posix()}?mode=ro", uri=True
                ) as connection:
                    row = connection.execute(
                        """
                        SELECT description, merchant FROM transactions
                        WHERE transaction_id = ? AND user_id = ?
                        """,
                        (transaction_id, user_id),
                    ).fetchone()
                if row is None:
                    raise HTTPException(status_code=404, detail="Transaction was not found.")
                description, merchant = row
            result = module.explain_category(
                description,
                merchant,
                database_path=DATABASE_PATH,
                model_dir=MODEL_DIRS["categorization"],
                output_dir=output_dir,
            )
            result_details = json.loads(
                Path(result["json_path"]).read_text(encoding="utf-8")
            )
            return {
                "status": "available",
                "model": model,
                "prediction_label": result["predicted_category"],
                "prediction_value": result_details["class_score"],
                "base_value": result_details["base_value"],
                "score_type": result_details["score_type"],
                "confidence": result_details["confidence"],
                "feature_contributions": [
                    {
                        "feature": row["term"],
                        "value": row["tfidf_value"],
                        "shap_value": row["shap_value"],
                    }
                    for row in result_details["top_term_contributions"]
                ],
                "explanation": result["explanation"],
            }

        if transaction_id:
            with sqlite3.connect(
                f"file:{DATABASE_PATH.resolve().as_posix()}?mode=ro",
                uri=True,
            ) as connection:
                owned_transaction = connection.execute(
                    """
                    SELECT 1 FROM transactions
                    WHERE transaction_id = ? AND user_id = ?
                    """,
                    (transaction_id, user_id),
                ).fetchone()
            if owned_transaction is None:
                raise HTTPException(status_code=404, detail="Transaction was not found.")
            result = module.explain_anomaly_transaction(
                transaction_id,
                database_path=DATABASE_PATH,
                model_dir=MODEL_DIRS["anomaly"],
                output_dir=output_dir,
            )
            return {
                "status": "available",
                "model": model,
                "prediction_label": result["prediction"],
                "prediction_value": result["anomaly_score"],
                "base_value": result["base_value"],
                "feature_contributions": result["feature_contributions"],
                "explanation": result["explanation"],
            }
        result = module.explain_anomaly_global(
            database_path=DATABASE_PATH,
            model_dir=MODEL_DIRS["anomaly"],
            output_dir=output_dir,
            scope="personal",
            user_id=user_id,
        )
        importance = result.get("mean_absolute_shap", {})
        return {
            "status": "available",
            "model": model,
            "feature_importance": [
                {"feature": feature, "importance": value}
                for feature, value in importance.items()
            ],
            "explanation": result.get("explanation"),
        }
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - reported to the client
        logger.exception("SHAP explanation could not be computed for model %s.", model)
        status = "stale" if "training cutoff" in str(exc).casefold() else "unavailable"
        return {"status": status, "reason": str(exc)}
