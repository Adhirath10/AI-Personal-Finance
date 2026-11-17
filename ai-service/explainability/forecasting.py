from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from explainability.common import (
    AI_SERVICE_DIR,
    deterministic_sample,
    explanation_rows,
    json_number,
    load_source_module,
    output_directory,
    require_shap,
    safe_filename,
    save_shap_plots,
    save_waterfall,
    write_global_importance,
    write_json,
)


DEFAULT_MODEL_DIR = AI_SERVICE_DIR / "models" / "forecasting"
_training = load_source_module(
    "_finance_xai_forecasting_training",
    AI_SERVICE_DIR / "forecasting" / "train.py",
)


def load_forecast_artifacts(
    model_dir: str | Path | None = None,
) -> tuple[Any, list[str]]:
    target = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_path = target / "forecasting_model.pkl"
    columns_path = target / "feature_columns.pkl"
    if not model_path.is_file() or not columns_path.is_file():
        raise FileNotFoundError(f"Forecasting artifacts were not found in {target}.")

    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Forecast feature_columns artifact must be a non-empty list.")
    if not all(isinstance(name, str) for name in feature_columns):
        raise ValueError("Forecast feature_columns artifact contains invalid names.")
    if getattr(model, "n_features_in_", None) != len(feature_columns):
        raise ValueError("Forecast model and feature_columns artifact do not match.")
    fitted_names = getattr(model, "feature_names_in_", None)
    if fitted_names is not None and list(fitted_names) != feature_columns:
        raise ValueError("Forecast model feature order differs from feature_columns.")
    return model, feature_columns


def _forecast_feature_rows(
    database_path: str | Path | None,
    feature_columns: list[str],
) -> pd.DataFrame:
    monthly = _training.build_monthly_expense_data(database_path)
    # Keep each user's latest month: it is the input for the actual next-month
    # forecast, so dropping it would explain a prediction for a month already recorded.
    dataset = _training.build_forecast_features(monthly)
    missing = [column for column in feature_columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"Forecast input data is missing features: {missing}.")
    return dataset


def _rows_through(dataset: pd.DataFrame, through_month: str | None) -> pd.DataFrame:
    """Keep feature rows up to ``through_month``; each row only uses earlier months."""
    if through_month is None:
        return dataset
    return dataset.loc[dataset["month"] <= pd.Period(through_month, freq="M")]


def _build_explainer(model: Any):
    shap = require_shap()
    return shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )


def explain_forecast_for_user(
    user_id: str,
    *,
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    through_month: str | None = None,
) -> dict[str, Any]:
    if not str(user_id).strip():
        raise ValueError("User ID cannot be empty.")
    model, feature_columns = load_forecast_artifacts(model_dir)
    dataset = _forecast_feature_rows(database_path, feature_columns)
    user_rows = _rows_through(
        dataset.loc[dataset["user_id"].astype(str) == str(user_id)],
        through_month,
    ).copy()
    if user_rows.empty:
        raise ValueError(f"User {user_id} has no eligible forecast history.")
    last_row = user_rows.sort_values("month").tail(1)
    feature_row = last_row[feature_columns].iloc[[0]]
    prediction = float(model.predict(feature_row)[0])
    forecast_month = (pd.Period(last_row["month"].iloc[0], freq="M") + 1).strftime(
        "%Y-%m"
    )

    values = _build_explainer(model)(feature_row)
    base_value = float(np.asarray(values.base_values).reshape(-1)[0])
    if not np.isclose(
        base_value + float(np.asarray(values.values).sum()),
        prediction,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise RuntimeError("Forecast SHAP values do not add up to the model output.")

    target_dir = output_directory(output_dir)
    stem = target_dir / f"forecast_user_{safe_filename(user_id)}"
    waterfall_path = save_waterfall(
        values,
        stem.with_name(f"{stem.name}_waterfall.png"),
        title=f"Forecast for {forecast_month}",
    )
    payload = {
        "model": "RandomForestRegressor",
        "user_id": str(user_id),
        "forecast_month": forecast_month,
        "prediction": prediction,
        "output_units": "same currency units as training target",
        "base_value": base_value,
        "feature_order": feature_columns,
        "feature_contributions": explanation_rows(values),
        "visualizations": [str(waterfall_path)],
        "explanation": (
            "SHAP values explain the Random Forest output relative to the "
            "tree-path reference. They are model attributions, not causal effects."
        ),
    }
    json_path = write_json(stem.with_suffix(".json"), payload)
    payload["json_path"] = str(json_path)
    return payload


def explain_forecast_global(
    *,
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    user_id: str | None = None,
    max_rows: int = 500,
    through_month: str | None = None,
) -> dict[str, Any]:
    model, feature_columns = load_forecast_artifacts(model_dir)
    dataset = _rows_through(
        _forecast_feature_rows(database_path, feature_columns), through_month
    )
    if user_id is not None:
        dataset = dataset.loc[dataset["user_id"].astype(str) == str(user_id)]
    if dataset.empty:
        raise ValueError("No forecast feature rows are available for this user.")
    features = deterministic_sample(dataset[feature_columns], max_rows)
    if features.empty:
        raise ValueError("No forecast feature rows are available to explain.")

    explanation = _build_explainer(model)(features)
    values = np.asarray(explanation.values)
    mean_absolute_values = np.abs(values).mean(axis=0)
    target_dir = output_directory(output_dir)
    stem = target_dir / "forecast_global"
    csv_path = write_global_importance(
        stem.with_suffix(".csv"),
        feature_columns,
        mean_absolute_values,
    )
    plots = save_shap_plots(
        explanation,
        stem,
        title="Expense forecast: global SHAP importance",
    )
    payload = {
        "model": "RandomForestRegressor",
        "explained_rows": int(len(features)),
        "feature_order": feature_columns,
        "mean_absolute_shap": {
            name: json_number(value)
            for name, value in zip(feature_columns, mean_absolute_values)
        },
        "global_importance_csv": str(csv_path),
        "visualizations": [str(path) for path in plots],
        "explanation": (
            "Global importance is mean absolute SHAP contribution over the "
            "available historical forecast input rows; it is descriptive, not causal."
        ),
    }
    json_path = write_json(stem.with_suffix(".json"), payload)
    payload["json_path"] = str(json_path)
    return payload
