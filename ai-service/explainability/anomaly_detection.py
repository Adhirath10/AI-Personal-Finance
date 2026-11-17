from __future__ import annotations

import importlib
import sqlite3
import sys
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


DEFAULT_MODEL_DIR = (
    AI_SERVICE_DIR / "models" / "anomaly_detection_behavioral_experiment"
)
_training = load_source_module(
    "_finance_xai_anomaly_training",
    AI_SERVICE_DIR / "anomaly_detection" / "train.py",
)


def _load_behavioral_feature_module():
    module_path = AI_SERVICE_DIR / "anomaly_detection" / "behavioral_experiment.py"
    existing = sys.modules.get("behavioral_experiment")
    if (
        existing is not None
        and Path(existing.__file__).resolve() == module_path.resolve()
    ):
        return existing

    script_dir = str(module_path.parent)
    aliases = ("train", "evaluate", "behavioral_experiment")
    previous_modules = {name: sys.modules.pop(name, None) for name in aliases}
    sys.path.insert(0, script_dir)
    try:
        return importlib.import_module("behavioral_experiment")
    finally:
        sys.path.remove(script_dir)
        for name in aliases:
            sys.modules.pop(name, None)
            if previous_modules[name] is not None:
                sys.modules[name] = previous_modules[name]


def load_anomaly_artifacts(
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_path = target / "behavioral_isolation_forest_models.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Behavioral anomaly artifacts were not found in {target}."
        )
    artifacts = joblib.load(model_path)
    required = {"pooled_model", "type_models", "user_models", "metadata"}
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError(f"Invalid behavioral anomaly artifacts in {model_path}.")
    if not artifacts["metadata"].get("training_end"):
        raise ValueError("Anomaly metadata does not contain a training_end cutoff.")
    return artifacts


def load_behavioral_transactions(
    database_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load feature-source columns only, through SQLite read-only mode."""
    path = _training.resolve_database_path(database_path)
    columns = [
        "transaction_id",
        "user_id",
        "transaction_date",
        "amount",
        "transaction_type",
        "category",
        "subcategory",
        "payment_mode",
        "account_type",
        "is_recurring",
        "balance_after_transaction",
        "merchant",
    ]
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "transactions" not in tables:
            raise ValueError("Table not found: transactions")
        available = {
            row[1] for row in connection.execute('PRAGMA table_info("transactions")')
        }
        missing = [column for column in columns if column not in available]
        if missing:
            raise ValueError(f"Missing behavioral feature columns: {missing}.")
        quoted = ", ".join(f'"{column}"' for column in columns)
        data = pd.read_sql_query(
            f'SELECT {quoted} FROM "transactions" WHERE "transaction_type" = ?',
            connection,
            params=("Expense",),
        )
    except sqlite3.Error as exc:
        raise sqlite3.DatabaseError(
            f"Unable to read behavioral feature inputs from {path}: {exc}"
        ) from exc
    finally:
        connection.close()

    if data.empty:
        raise ValueError("No expense transactions are available for explanation.")
    data = data.copy()
    data["transaction_date"] = pd.to_datetime(data["transaction_date"], errors="coerce")
    if data["transaction_date"].isna().any():
        raise ValueError("Behavioral explanations require valid transaction dates.")
    return data


def _select_model(
    artifacts: dict[str, Any],
    user_id: str,
    transaction_type: str = "Expense",
    *,
    scope: str | None = None,
):
    personal_key = (str(user_id), str(transaction_type))
    if scope in (None, "personal") and personal_key in artifacts["user_models"]:
        return artifacts["user_models"][personal_key], "personal"
    if scope == "personal":
        raise ValueError(
            f"No personal anomaly model exists for user {user_id} ({transaction_type})."
        )
    if scope in (None, "type") and transaction_type in artifacts["type_models"]:
        return artifacts["type_models"][transaction_type], "type"
    if scope == "type":
        raise ValueError(
            f"No anomaly model exists for transaction type {transaction_type}."
        )
    if scope not in (None, "pooled"):
        raise ValueError("scope must be one of: personal, type, pooled.")
    return artifacts["pooled_model"], "pooled"


def _prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    module = _load_behavioral_feature_module()
    return module.prepare_experiment_features(data)


def _validate_artifact_features(
    artifacts: dict[str, Any],
    model: Any,
    features: pd.DataFrame,
) -> None:
    expected = artifacts["metadata"].get("feature_columns")
    if not isinstance(expected, list) or set(expected) != set(features.columns):
        raise ValueError(
            "Behavioral artifact feature metadata differs from the builder."
        )
    preprocessor = model.named_steps["preprocessor"]
    fitted_columns = list(getattr(preprocessor, "feature_names_in_", []))
    if fitted_columns and fitted_columns != list(features.columns):
        raise ValueError(
            "Behavioral feature builder order differs from the fitted pipeline input."
        )


def _background_features(
    data: pd.DataFrame,
    artifacts: dict[str, Any],
    model_scope: str,
    user_id: str | None,
    transaction_type: str,
) -> pd.DataFrame:
    training_end = pd.Timestamp(artifacts["metadata"]["training_end"])
    training_rows = data.loc[data["transaction_date"] <= training_end].copy()
    expected_rows = artifacts["metadata"].get("training_rows")
    if expected_rows is not None and len(training_rows) != int(expected_rows):
        raise ValueError(
            "The database's pre-cutoff row count differs from the saved model's "
            "training_rows; refusing to use an unverified SHAP background."
        )
    if model_scope == "personal":
        training_rows = training_rows.loc[
            (training_rows["user_id"].astype(str) == str(user_id))
            & (training_rows["transaction_type"].astype(str) == transaction_type)
        ]
    elif model_scope == "type":
        training_rows = training_rows.loc[
            training_rows["transaction_type"].astype(str) == transaction_type
        ]

    if training_rows.empty:
        raise ValueError(
            "No training-period rows are available for the SHAP background."
        )
    return _prepare_features(training_rows)


def _score_function(model: Any, feature_columns: list[str]):
    def score(data):
        if isinstance(data, pd.DataFrame):
            frame = data.loc[:, feature_columns]
        else:
            frame = pd.DataFrame(data, columns=feature_columns)
        return -np.asarray(model.decision_function(frame), dtype=float).reshape(-1)

    return score


def _explain_rows(
    model: Any,
    background: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    max_evals: int | None = None,
):
    shap = require_shap()
    feature_columns = list(rows.columns)
    categorical_columns = list(
        rows.select_dtypes(include=["object", "string", "category"]).columns
    )
    mappings = {
        column: {
            value: index
            for index, value in enumerate(
                sorted(
                    set(background[column].astype(str)) | set(rows[column].astype(str))
                )
            )
        }
        for column in categorical_columns
    }

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.loc[:, feature_columns].copy()
        for column, mapping in mappings.items():
            encoded[column] = encoded[column].astype(str).map(mapping).astype(float)
        return encoded

    def score(masked):
        masked_frame = pd.DataFrame(
            np.asarray(masked).reshape(-1, len(feature_columns)),
            columns=feature_columns,
        )
        for column, mapping in mappings.items():
            reverse_mapping = {value: key for key, value in mapping.items()}
            codes = np.rint(masked_frame[column]).astype(int)
            masked_frame[column] = codes.map(reverse_mapping)
        return _score_function(model, feature_columns)(masked_frame)

    encoded_background = encode(background)
    encoded_rows = encode(rows)
    masker = shap.maskers.Independent(
        encoded_background.to_numpy(dtype=float),
        max_samples=len(encoded_background),
    )
    explainer = shap.Explainer(
        score,
        masker,
        algorithm="permutation",
        feature_names=feature_columns,
    )
    evaluations = max_evals or 3 * (2 * len(feature_columns) + 1)
    masked_explanation = explainer(
        encoded_rows.to_numpy(dtype=float),
        max_evals=evaluations,
    )
    return shap.Explanation(
        values=masked_explanation.values,
        base_values=masked_explanation.base_values,
        data=rows.to_numpy(dtype=object),
        feature_names=feature_columns,
    )


def explain_anomaly_transaction(
    transaction_id: str,
    *,
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    background_size: int = 50,
) -> dict[str, Any]:
    if not str(transaction_id).strip():
        raise ValueError("Transaction ID cannot be empty.")
    if background_size < 1:
        raise ValueError("background_size must be greater than zero.")
    artifacts = load_anomaly_artifacts(model_dir)
    data = load_behavioral_transactions(database_path)
    selected_rows = data.loc[data["transaction_id"].astype(str) == str(transaction_id)]
    if selected_rows.empty:
        raise ValueError(f"Expense transaction not found: {transaction_id}")
    target = selected_rows.iloc[0]
    training_end = pd.Timestamp(artifacts["metadata"]["training_end"])
    if target["transaction_date"] <= training_end:
        raise ValueError(
            "The behavioral model is intended to score transactions after its "
            f"training cutoff ({training_end.date()})."
        )

    history = data.loc[
        (data["user_id"].astype(str) == str(target["user_id"]))
        & (data["transaction_date"] <= target["transaction_date"])
    ].copy()
    features = _prepare_features(history)
    feature_row = features.loc[
        history["transaction_id"].astype(str) == str(transaction_id)
    ]
    model, model_scope = _select_model(
        artifacts,
        str(target["user_id"]),
        str(target["transaction_type"]),
    )
    _validate_artifact_features(artifacts, model, features)
    score = float(-model.decision_function(feature_row)[0])
    prediction = int(model.predict(feature_row)[0] == -1)

    background = _background_features(
        data,
        artifacts,
        model_scope,
        str(target["user_id"]),
        str(target["transaction_type"]),
    )
    background = deterministic_sample(background, background_size)
    explanation = _explain_rows(model, background, feature_row)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    if not np.isclose(
        base_value + float(np.asarray(explanation.values).sum()),
        score,
        rtol=2e-2,
        atol=2e-3,
    ):
        raise RuntimeError("Anomaly SHAP values do not add up to the anomaly score.")

    target_dir = output_directory(output_dir)
    stem = target_dir / f"anomaly_{safe_filename(transaction_id)}"
    waterfall_path = save_waterfall(
        explanation,
        stem.with_name(f"{stem.name}_waterfall.png"),
        title="Isolation Forest anomaly score",
    )
    payload = {
        "model": "IsolationForest",
        "model_scope": model_scope,
        "transaction_id": str(transaction_id),
        "transaction_date": target["transaction_date"].date().isoformat(),
        "training_end": training_end.date().isoformat(),
        "prediction": "Potential Anomaly" if prediction else "Normal",
        "anomaly_score": score,
        "score_direction": "higher means more unusual",
        "base_value": base_value,
        "feature_contributions": explanation_rows(explanation),
        "visualizations": [str(waterfall_path)],
        "explanation": (
            "Permutation SHAP approximates feature contributions to the negated "
            "Isolation Forest decision_function. The score is not a fraud probability "
            "or a causal explanation."
        ),
    }
    json_path = write_json(stem.with_suffix(".json"), payload)
    payload["json_path"] = str(json_path)
    return payload


def explain_anomaly_global(
    *,
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    scope: str = "pooled",
    user_id: str | None = None,
    transaction_type: str = "Expense",
    background_size: int = 50,
    max_rows: int = 100,
) -> dict[str, Any]:
    if scope not in {"personal", "type", "pooled"}:
        raise ValueError("scope must be one of: personal, type, pooled.")
    if scope == "personal" and not user_id:
        raise ValueError("user_id is required for a personal-model global explanation.")
    if background_size < 1 or max_rows < 1:
        raise ValueError("background_size and max_rows must be greater than zero.")

    artifacts = load_anomaly_artifacts(model_dir)
    data = load_behavioral_transactions(database_path)
    training_end = pd.Timestamp(artifacts["metadata"]["training_end"])
    model, model_scope = _select_model(
        artifacts,
        str(user_id or ""),
        transaction_type,
        scope=scope,
    )
    _validate_artifact_features(
        artifacts,
        model,
        _prepare_features(data.head(1)),
    )

    features = _prepare_features(data)
    eligible = data["transaction_date"] > training_end
    eligible &= data["transaction_type"].astype(str) == transaction_type
    personal_keys = artifacts["user_models"]
    if model_scope == "personal":
        eligible &= data["user_id"].astype(str) == str(user_id)
    elif model_scope == "type":
        eligible &= ~data.apply(
            lambda row: (str(row["user_id"]), transaction_type) in personal_keys,
            axis=1,
        )
    else:
        eligible &= ~data.apply(
            lambda row: (str(row["user_id"]), transaction_type) in personal_keys
            or transaction_type in artifacts["type_models"],
            axis=1,
        )
    candidate_features = features.loc[eligible]
    if candidate_features.empty:
        raise ValueError(
            f"No out-of-sample transactions use the requested {model_scope} model."
        )
    candidate_features = deterministic_sample(candidate_features, max_rows)
    background = _background_features(
        data,
        artifacts,
        model_scope,
        user_id,
        transaction_type,
    )
    background = deterministic_sample(background, background_size)
    explanation = _explain_rows(model, background, candidate_features)
    values = np.asarray(explanation.values)
    if values.ndim != 2:
        raise ValueError("Expected single-output anomaly score explanations.")
    mean_absolute_values = np.abs(values).mean(axis=0)
    feature_columns = list(candidate_features.columns)

    target_dir = output_directory(output_dir)
    scope_name = (
        f"anomaly_global_{model_scope}_{safe_filename(user_id)}"
        if model_scope == "personal"
        else f"anomaly_global_{model_scope}_{safe_filename(transaction_type)}"
    )
    stem = target_dir / scope_name
    csv_path = write_global_importance(
        stem.with_suffix(".csv"),
        feature_columns,
        mean_absolute_values,
    )
    plots = save_shap_plots(
        explanation,
        stem,
        title=f"Isolation Forest ({model_scope}): global SHAP importance",
    )
    payload = {
        "model": "IsolationForest",
        "model_scope": model_scope,
        "transaction_type": transaction_type,
        "user_id": str(user_id) if model_scope == "personal" else None,
        "training_end": training_end.date().isoformat(),
        "explained_rows": int(len(candidate_features)),
        "background_rows": int(len(background)),
        "score_direction": "higher means more unusual",
        "mean_absolute_shap": {
            name: json_number(value)
            for name, value in zip(feature_columns, mean_absolute_values)
        },
        "global_importance_csv": str(csv_path),
        "visualizations": [str(path) for path in plots],
        "explanation": (
            "Global importance is mean absolute permutation SHAP on eligible "
            "out-of-sample rows for this model scope. It explains model score, "
            "not fraud probability or causality."
        ),
    }
    json_path = write_json(stem.with_suffix(".json"), payload)
    payload["json_path"] = str(json_path)
    return payload
