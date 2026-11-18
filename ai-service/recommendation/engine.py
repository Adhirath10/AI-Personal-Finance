from __future__ import annotations

import logging
import sqlite3
import sys
import threading
from collections import Counter
from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "personal_finance.db"
DEFAULT_FORECAST_MODEL_DIR = PROJECT_ROOT / "ai-service" / "models" / "forecasting"
DEFAULT_CATEGORIZATION_MODEL_DIR = (
    PROJECT_ROOT / "ai-service" / "models" / "categorization"
)
DEFAULT_ANOMALY_MODEL_DIR = (
    PROJECT_ROOT
    / "ai-service"
    / "models"
    / "anomaly_detection_behavioral_experiment"
)

TRANSACTION_COLUMNS = [
    "transaction_id",
    "user_id",
    "transaction_date",
    "description",
    "merchant",
    "amount",
    "transaction_type",
    "category",
    "subcategory",
    "payment_mode",
    "account_type",
    "is_recurring",
    "balance_after_transaction",
]
SUMMARY_COLUMNS = [
    "summary_month",
    "total_income",
    "total_expense",
    "total_savings",
    "largest_category",
    "largest_category_amount",
    "anomaly_count",
]
BUDGET_COLUMNS = [
    "budget_month",
    "category",
    "budget_amount",
    "actual_amount",
    "variance_amount",
    "exceeded",
]
GOAL_COLUMNS = [
    "goal_id",
    "goal_type",
    "target_amount",
    "current_amount",
    "target_date",
    "priority",
    "status",
]
USER_COLUMNS = ["monthly_income"]
INVESTMENT_CATEGORIES = {"investment", "investments"}
DISCRETIONARY_CATEGORIES = {
    "dining",
    "entertainment",
    "food",
    "personal care",
    "shopping",
    "subscription",
    "travel",
}


# Saved models are loaded once per file version and shared by all requests.
# Loading is serialised: concurrent first-time loads import the same
# scikit-learn modules from several threads, which can deadlock Python's
# import system (seen when the dashboard runs all AI models at once).
_ARTIFACT_LOCK = threading.RLock()
_ARTIFACT_CACHE: dict[tuple[str, int], Any] = {}


def _load_artifact(path: Path) -> Any:
    resolved = Path(path).resolve()
    key = (str(resolved), resolved.stat().st_mtime_ns)
    with _ARTIFACT_LOCK:
        if key not in _ARTIFACT_CACHE:
            _ARTIFACT_CACHE[key] = joblib.load(resolved)
        return _ARTIFACT_CACHE[key]


_FORECAST_TRAINING = None


def _forecasting_training_module():
    """The forecasting training module, loaded once (it holds the shared feature builder)."""
    global _FORECAST_TRAINING
    with _ARTIFACT_LOCK:
        if _FORECAST_TRAINING is None:
            path = PROJECT_ROOT / "ai-service" / "forecasting" / "train.py"
            spec = spec_from_file_location("_finai_forecasting_training", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load forecasting helpers from {path}.")
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            _FORECAST_TRAINING = module
        return _FORECAST_TRAINING


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _month(value: object) -> pd.Period | None:
    try:
        if pd.isna(value):
            return None
        return pd.Period(str(value), freq="M")
    except (TypeError, ValueError):
        return None


def _clean_category(value: object) -> str | None:
    if pd.isna(value):
        return None
    category = str(value).strip()
    return category or None


def _read_user_table(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    user_id: str,
    *,
    required: bool = False,
) -> pd.DataFrame:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if table not in tables:
        message = f"Database table {table!r} is unavailable."
        if required:
            raise ValueError(message)
        LOGGER.warning("%s Related recommendations may be omitted.", message)
        return _empty_frame(columns)

    available = {
        row[1] for row in connection.execute(f"PRAGMA table_info({_quoted(table)})")
    }
    missing = [column for column in columns + ["user_id"] if column not in available]
    if missing:
        message = f"Table {table!r} is missing required columns: {missing}."
        if required:
            raise ValueError(message)
        LOGGER.warning("%s Related recommendations may be omitted.", message)
        return _empty_frame(columns)

    selected = ", ".join(_quoted(column) for column in columns)
    query = (
        f"SELECT {selected} FROM {_quoted(table)} "
        f"WHERE {_quoted('user_id')} = ?"
    )
    return pd.read_sql_query(query, connection, params=(user_id,))


def load_user_financial_data(
    database_path: str | Path | None,
    user_id: str,
) -> dict[str, pd.DataFrame]:
    """Load a single user's finance data using a read-only SQLite connection."""
    path = Path(database_path).expanduser() if database_path else DEFAULT_DATABASE_PATH
    resolved = path if path.is_absolute() else Path.cwd() / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Finance database not found: {resolved}")

    try:
        connection = sqlite3.connect(
            f"file:{resolved.as_posix()}?mode=ro",
            uri=True,
        )
    except sqlite3.Error as exc:
        raise sqlite3.OperationalError(
            f"Unable to open finance database read-only: {resolved}: {exc}"
        ) from exc

    try:
        result = {
            "transactions": _read_user_table(
                connection,
                "transactions",
                TRANSACTION_COLUMNS,
                user_id,
                required=True,
            ),
            "summaries": _read_user_table(
                connection, "monthly_financial_summary", SUMMARY_COLUMNS, user_id
            ),
            "budgets": _read_user_table(
                connection, "monthly_budgets", BUDGET_COLUMNS, user_id
            ),
            "goals": _read_user_table(
                connection, "financial_goals", GOAL_COLUMNS, user_id
            ),
            "users": _read_user_table(
                connection, "users", USER_COLUMNS, user_id
            ),
        }
    except sqlite3.Error as exc:
        raise sqlite3.DatabaseError(
            f"Unable to read finance data for user {user_id!r}: {exc}"
        ) from exc
    finally:
        connection.close()

    if result["transactions"].empty:
        raise ValueError(f"No transactions found for user ID {user_id!r}.")
    return result


def _inr(amount: float) -> str:
    """Format rupees with Indian digit grouping, e.g. 125000 -> ₹1,25,000."""
    whole = str(int(round(abs(amount))))
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return ("-" if amount < 0 else "") + "₹" + whole


def _typical_monthly_income(
    rows: pd.DataFrame, kinds: pd.Series, period: pd.Period | None
) -> float | None:
    """Median monthly income over the six recorded months up to the period."""
    income = rows.loc[kinds.eq("income")]
    if income.empty:
        return None
    monthly = income.groupby(income["transaction_date"].dt.to_period("M"))["amount"].sum()
    if period is not None:
        monthly = monthly[monthly.index <= period]
    monthly = monthly.tail(6)
    value = float(monthly.median()) if not monthly.empty else None
    return value if value and value > 0 else None


# Transparent checks against the user's own earlier expenses.
LARGE_PURCHASE_RATIO = 3.0  # this many times the usual purchase in the category
VERY_LARGE_PURCHASE_RATIO = 5.0
EXTREME_PURCHASE_RATIO = 8.0
DUPLICATE_WINDOW_DAYS = 3
MIN_CATEGORY_HISTORY = 5
MIN_MERCHANT_HISTORY = 20


def _assess_expense(
    row: pd.Series,
    index: Any,
    expense_rows: pd.DataFrame,
    typical_income: float | None,
    model_flag: bool,
    score: float,
) -> dict[str, Any] | None:
    """Return severity and plain-language reasons for an unusual expense, or None."""
    amount = float(row["amount"])
    when = row["transaction_date"]
    category = _clean_category(row["category"]) or "this category"
    merchant = str(row["merchant"]).strip() if pd.notna(row["merchant"]) else ""
    position = expense_rows.index.get_loc(index)
    prior = expense_rows.iloc[:position]
    prior = prior.loc[prior["transaction_date"] <= when]

    reasons: list[str] = []
    signals: list[str] = []
    points = 0

    same_category = prior.loc[prior["category"].map(_clean_category) == _clean_category(row["category"]), "amount"]
    usual = float(same_category.median()) if len(same_category) >= MIN_CATEGORY_HISTORY else None
    ratio = amount / usual if usual else None
    if ratio is not None and ratio >= LARGE_PURCHASE_RATIO:
        points += 4 if ratio >= EXTREME_PURCHASE_RATIO else 3 if ratio >= VERY_LARGE_PURCHASE_RATIO else 2
        signals.append("large_for_category")
        reasons.append(
            f"{_inr(amount)} is {ratio:.1f}× your usual {category} purchase ({_inr(usual)})."
        )

    window = prior.loc[prior["transaction_date"] >= when - pd.Timedelta(days=DUPLICATE_WINDOW_DAYS)]
    duplicates = window.loc[
        window["merchant"].astype(str).str.strip().str.casefold().eq(merchant.casefold())
        & (window["amount"] - amount).abs().lt(0.01)
    ]
    if merchant and not duplicates.empty:
        points += 3
        signals.append("possible_duplicate")
        earlier = duplicates["transaction_date"].iloc[-1]
        days = (when.normalize() - earlier.normalize()).days
        reasons.append(
            f"{merchant} charged the same {_inr(amount)} "
            + ("earlier the same day" if days == 0 else f"{days} day{'s' if days != 1 else ''} earlier")
            + ": possible duplicate charge."
        )

    if typical_income:
        share = amount / typical_income
        if share >= 0.25:
            points += 2
            signals.append("large_share_of_income")
            reasons.append(f"Uses {share:.0%} of your typical monthly income.")
        elif share >= 0.10 and ratio is not None and ratio >= 2:
            points += 1
            signals.append("large_share_of_income")
            reasons.append(f"Uses {share:.0%} of your typical monthly income.")

    known_merchants = set(prior["merchant"].astype(str).str.strip().str.casefold())
    if merchant and len(prior) >= MIN_MERCHANT_HISTORY and merchant.casefold() not in known_merchants:
        points += 1
        signals.append("new_merchant")
        reasons.append(f"First payment to {merchant}.")

    if model_flag:
        # The model alone only marks a pattern for review (Low); it raises the
        # severity when one of the checks above agrees.
        points += 2 if score >= 0.05 and signals else 1
        signals.append("model")
        reasons.append(
            f"The anomaly model finds this unusual for your {category} spending "
            f"(timing, amount or frequency; score {score:.3f})."
        )

    if not model_flag and points < 2:
        return None
    severity = "High" if points >= 4 else "Medium" if points >= 2 else "Low"
    rule_based = [signal for signal in signals if signal != "model"]
    return {
        "severity": severity,
        "risk_points": points,
        "reasons": reasons,
        "signals": signals,
        "usual_category_amount": usual,
        "amount_ratio": ratio,
        "detected_by": (
            "model + rules" if model_flag and rule_based else "model" if model_flag else "rules"
        ),
    }


class RecommendationEngine:
    """Create personalized recommendations without changing source data or models."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        forecast_model_dir: str | Path | None = None,
        categorization_model_dir: str | Path | None = None,
        anomaly_model_dir: str | Path | None = None,
        today: date | None = None,
        analysis_month: str | pd.Period | None = None,
        category_increase_threshold: float = 0.10,
        budget_warning_threshold: float = 0.80,
    ) -> None:
        if not 0 <= category_increase_threshold:
            raise ValueError("category_increase_threshold must be non-negative.")
        if not 0 < budget_warning_threshold <= 1:
            raise ValueError("budget_warning_threshold must be in (0, 1].")
        self.database_path = database_path or DEFAULT_DATABASE_PATH
        self.forecast_model_dir = Path(
            forecast_model_dir or DEFAULT_FORECAST_MODEL_DIR
        )
        self.categorization_model_dir = Path(
            categorization_model_dir or DEFAULT_CATEGORIZATION_MODEL_DIR
        )
        self.anomaly_model_dir = Path(anomaly_model_dir or DEFAULT_ANOMALY_MODEL_DIR)
        self.today = today or date.today()
        self.analysis_month = self._parse_analysis_month(analysis_month)
        self.category_increase_threshold = category_increase_threshold
        self.budget_warning_threshold = budget_warning_threshold
        self._categorization_artifacts: tuple[Any, Any] | None = None
        self._forecast_artifacts: tuple[Any, list[str]] | None = None
        self._forecast_metadata: dict | None = None
        self._forecast_training = None
        self._forecast_unavailable_reason: str | None = None
        self._anomaly_artifacts: dict | None = None

    @staticmethod
    def _parse_analysis_month(value: str | pd.Period | None) -> pd.Period | None:
        if value is None:
            return None
        try:
            period = pd.Period(value, freq="M")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "analysis_month must use YYYY-MM format."
            ) from exc
        if str(value) != str(period):
            raise ValueError("analysis_month must use YYYY-MM format.")
        return period

    @staticmethod
    def _is_investment_category(category: object) -> bool:
        normalized = str(category).strip().casefold() if pd.notna(category) else ""
        return normalized in INVESTMENT_CATEGORIES

    @staticmethod
    def _is_discretionary_category(category: object) -> bool:
        normalized = str(category).strip().casefold() if pd.notna(category) else ""
        return normalized in DISCRETIONARY_CATEGORIES

    def _load_categorization_artifacts(self) -> tuple[Any, Any] | None:
        if self._categorization_artifacts is not None:
            return self._categorization_artifacts
        vectorizer_path = self.categorization_model_dir / "tfidf_vectorizer.pkl"
        classifier_path = self.categorization_model_dir / "category_classifier.pkl"
        if not vectorizer_path.is_file() or not classifier_path.is_file():
            LOGGER.warning(
                "Categorization model artifacts are unavailable in %s; uncategorized "
                "transactions will be excluded from category-specific analysis.",
                self.categorization_model_dir,
            )
            return None
        try:
            vectorizer = _load_artifact(vectorizer_path)
            classifier = _load_artifact(classifier_path)
        except (OSError, EOFError, ValueError, ImportError) as exc:
            raise ValueError(
                f"Unable to load categorization model artifacts: {exc}"
            ) from exc
        self._categorization_artifacts = (vectorizer, classifier)
        return self._categorization_artifacts

    def _predict_missing_categories(
        self, transactions: pd.DataFrame
    ) -> tuple[pd.Series, dict[str, Any]]:
        categories = transactions["category"].map(_clean_category)
        missing_mask = categories.isna()
        if not missing_mask.any():
            return categories, {
                "status": "not_needed",
                "transactions_categorized": 0,
            }

        artifacts = self._load_categorization_artifacts()
        if artifacts is None:
            return categories, {
                "status": "unavailable",
                "transactions_categorized": 0,
                "uncategorized_transactions": int(missing_mask.sum()),
            }

        categorization_train_path = (
            PROJECT_ROOT / "ai-service" / "categorization" / "train.py"
        )
        spec = spec_from_file_location(
            "recommendation_categorization_training", categorization_train_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to load text preparation helper from {categorization_train_path}."
            )
        categorization_training = module_from_spec(spec)
        try:
            with _ARTIFACT_LOCK:
                spec.loader.exec_module(categorization_training)
        except (ImportError, OSError, SyntaxError) as exc:
            raise RuntimeError(
                "Unable to import the existing transaction text preparation helper."
            ) from exc

        vectorizer, classifier = artifacts
        missing_rows = transactions.loc[missing_mask]
        text = [
            categorization_training.prepare_text_feature(row.description, row.merchant)
            for row in missing_rows.itertuples(index=False)
        ]
        predictions = classifier.predict(vectorizer.transform(text))
        categories.loc[missing_mask] = pd.Series(
            predictions,
            index=missing_rows.index,
            dtype="object",
        )
        LOGGER.info(
            "Used the existing categorization model for %d uncategorized transaction(s).",
            int(missing_mask.sum()),
        )
        return categories, {
            "status": "used",
            "transactions_categorized": int(missing_mask.sum()),
        }

    def _load_forecast_artifacts(self) -> tuple[Any, list[str]] | None:
        if self._forecast_artifacts is not None:
            return self._forecast_artifacts
        model_path = self.forecast_model_dir / "forecasting_model.pkl"
        columns_path = self.forecast_model_dir / "feature_columns.pkl"
        if not model_path.is_file() or not columns_path.is_file():
            self._forecast_unavailable_reason = (
                f"Forecast model artifacts are missing from {self.forecast_model_dir}."
            )
            LOGGER.warning(
                "Forecast model artifacts are unavailable in %s; future-expense "
                "recommendations will be omitted.",
                self.forecast_model_dir,
            )
            return None
        try:
            model = _load_artifact(model_path)
            feature_columns = _load_artifact(columns_path)
        except (OSError, EOFError, ValueError, ImportError) as exc:
            raise ValueError(f"Unable to load forecast model artifacts: {exc}") from exc
        if (
            not isinstance(feature_columns, list)
            or not feature_columns
            or not all(isinstance(column, str) for column in feature_columns)
        ):
            raise ValueError("Forecast model feature_columns artifact is invalid.")
        if not callable(getattr(model, "predict", None)):
            self._forecast_unavailable_reason = (
                "Forecast artifact does not expose a prediction interface."
            )
            LOGGER.warning("Forecast artifact does not expose a prediction interface.")
            return None
        try:
            check_is_fitted(model)
        except NotFittedError:
            self._forecast_unavailable_reason = (
                "Forecast model artifact has not been fitted."
            )
            LOGGER.warning("Forecast artifact exists but is not a fitted model.")
            return None
        except TypeError as exc:
            self._forecast_unavailable_reason = (
                f"Forecast artifact is not a supported fitted estimator: {exc}"
            )
            LOGGER.warning("%s", self._forecast_unavailable_reason)
            return None
        expected_features = getattr(model, "n_features_in_", len(feature_columns))
        if expected_features != len(feature_columns):
            self._forecast_unavailable_reason = (
                f"Forecast artifact expects {expected_features} feature(s), but "
                f"metadata lists {len(feature_columns)}."
            )
            LOGGER.warning(
                "Forecast artifact expects %s feature(s), but metadata lists %s.",
                expected_features,
                len(feature_columns),
            )
            return None
        self._forecast_artifacts = (model, feature_columns)
        self._forecast_training = _forecasting_training_module()
        self._forecast_metadata = self._forecast_training.load_forecast_metadata(self.forecast_model_dir)
        return self._forecast_artifacts

    def _forecast_user_expenses(
        self,
        transactions: pd.DataFrame,
        profile: pd.DataFrame,
        analysis_period: pd.Period,
    ) -> dict[str, Any]:
        artifacts = self._load_forecast_artifacts()
        if artifacts is None:
            return {
                "status": "unavailable",
                "skip_reason": self._forecast_unavailable_reason
                or "Forecast model artifacts are unavailable.",
            }

        dated = transactions.copy()
        dated["transaction_date"] = pd.to_datetime(
            dated["transaction_date"], errors="coerce"
        )
        dated["amount"] = pd.to_numeric(dated["amount"], errors="coerce")
        expense_mask = dated["transaction_type"].astype(str).str.casefold().eq("expense")
        expenses = dated.loc[expense_mask].dropna(
            subset=["transaction_date", "amount"]
        )
        expenses = expenses.loc[
            expenses["transaction_date"].dt.to_period("M") <= analysis_period
        ].copy()
        if expenses.empty:
            return {
                "status": "insufficient_history",
                "skip_reason": (
                    f"No expense transactions are available through {analysis_period}."
                ),
            }
        expenses["month"] = expenses["transaction_date"].dt.to_period("M")
        monthly = (
            expenses.groupby("month")["amount"]
            .sum()
            .sort_index()
        )
        if len(monthly) < 3:
            return {
                "status": "insufficient_history",
                "skip_reason": "At least three recorded expense months are required.",
                "available_months": int(len(monthly)),
                "required_months": 3,
            }
        latest_history_month = monthly.index[-1]
        if latest_history_month != analysis_period:
            return {
                "status": "insufficient_history",
                "skip_reason": (
                    f"The selected analysis month {analysis_period} has no "
                    "expense observations; a forecast for that period cannot be "
                    "verified against the selected data."
                ),
                "latest_expense_month": str(latest_history_month),
            }

        model, feature_columns = artifacts
        latest_month = analysis_period
        training = self._forecast_training
        # Same feature builder as training, so features cannot drift apart.
        history = pd.DataFrame({
            "user_id": "user",
            "month": [str(period) for period in monthly.index],
            "total_expense": monthly.to_numpy(dtype=float),
        })
        feature_row = training.build_forecast_features(history).tail(1)
        missing_features = [name for name in feature_columns if name not in feature_row.columns]
        if missing_features:
            LOGGER.warning(
                "Forecast model requests unsupported features: %s", missing_features
            )
            return {
                "status": "unavailable",
                "skip_reason": (
                    "Forecast feature metadata contains unsupported columns: "
                    f"{missing_features}."
                ),
            }
        try:
            forecast = float(
                training.predict_expense(
                    model, self._forecast_metadata, feature_row, feature_columns
                )[0]
            )
        except (ValueError, TypeError, AttributeError) as exc:
            LOGGER.exception(
                "Forecast prediction failed for analysis month %s.", analysis_period
            )
            return {
                "status": "unavailable",
                "skip_reason": f"Forecast prediction failed: {exc}",
            }
        forecast_month = latest_month + 1
        # The current month is still in progress, so forecasting it is useful;
        # only months that have already ended are stale.
        is_stale = forecast_month < pd.Period(self.today, freq="M")
        if not np.isfinite(forecast):
            return {
                "status": "unavailable",
                "skip_reason": "Forecast model returned a non-finite value.",
                "forecast_month": str(forecast_month),
            }
        monthly_income = (
            _number(profile.iloc[0]["monthly_income"])
            if not profile.empty
            else None
        )
        return {
            "status": "stale" if is_stale else "available",
            "forecast_month": str(forecast_month),
            "predicted_expense": forecast,
            "monthly_income_profile": monthly_income,
            "history_months": int(len(monthly)),
            "skip_reason": (
                f"Forecast month {forecast_month} is not later than the "
                f"current month {pd.Period(self.today, freq='M')}."
                if is_stale
                else None
            ),
        }

    def _load_anomaly_artifacts(self) -> dict | None:
        if self._anomaly_artifacts is not None:
            return self._anomaly_artifacts
        model_path = self.anomaly_model_dir / "behavioral_isolation_forest_models.joblib"
        if not model_path.is_file():
            LOGGER.warning(
                "Behavioral anomaly model is unavailable at %s; unusual-transaction "
                "recommendations will be omitted.",
                model_path,
            )
            return None
        try:
            artifacts = _load_artifact(model_path)
        except (OSError, EOFError, ValueError, ImportError) as exc:
            raise ValueError(f"Unable to load behavioral anomaly model: {exc}") from exc
        required = {"pooled_model", "type_models", "user_models", "metadata"}
        if not isinstance(artifacts, dict) or not required.issubset(artifacts):
            raise ValueError(f"Invalid behavioral anomaly artifacts at {model_path}.")
        self._anomaly_artifacts = artifacts
        return artifacts

    def _score_potential_anomalies(
        self,
        transactions: pd.DataFrame,
        analysis_period: pd.Period | None = None,
    ) -> dict[str, Any]:
        """Flag unusual expenses in the analysis month and explain why.

        The saved behavioral Isolation Forest scores each out-of-sample expense.
        Transparent checks against the user's own earlier history (amount
        versus their usual purchase in the category, first payment to a
        merchant, share of monthly income, a repeated identical charge) explain
        the flags, set their severity, and catch clear cases the model misses.
        """
        artifacts = self._load_anomaly_artifacts()
        if artifacts is None:
            return {"status": "unavailable", "transactions": []}
        metadata = artifacts["metadata"]
        training_end = pd.to_datetime(metadata.get("training_end"), errors="coerce")
        if pd.isna(training_end):
            raise ValueError("Anomaly model metadata has no valid training_end date.")

        anomaly_dir = PROJECT_ROOT / "ai-service" / "anomaly_detection"
        for directory in (anomaly_dir,):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        try:
            with _ARTIFACT_LOCK:
                from behavioral_experiment import prepare_experiment_features
        except ImportError as exc:
            raise RuntimeError(
                "Unable to import behavioral anomaly feature preparation."
            ) from exc

        rows = transactions.copy()
        rows["transaction_date"] = pd.to_datetime(
            rows["transaction_date"], errors="coerce"
        )
        kinds = rows["transaction_type"].astype(str).str.strip().str.casefold()
        out_of_sample = rows["transaction_date"].gt(training_end) & kinds.eq("expense")
        if analysis_period is not None:
            out_of_sample &= rows["transaction_date"].dt.to_period("M").eq(analysis_period)
        scope = {
            "training_end": training_end.date().isoformat(),
            "scope_month": str(analysis_period) if analysis_period is not None else None,
        }
        if not out_of_sample.any():
            return {
                "status": "no_out_of_sample_rows",
                **scope,
                "analyzed_count": 0,
                "transactions": [],
            }

        features = prepare_experiment_features(rows)
        candidates = rows.loc[out_of_sample]
        candidate_features = features.loc[candidates.index]
        model_results: dict[Any, tuple[bool, float]] = {}
        for (user_id, transaction_type), indexes in candidates.groupby(
            ["user_id", "transaction_type"], sort=False
        ).groups.items():
            model = artifacts["user_models"].get(
                (str(user_id), str(transaction_type)),
                artifacts["type_models"].get(
                    str(transaction_type), artifacts["pooled_model"]
                ),
            )
            selected = candidate_features.loc[indexes]
            flags = model.predict(selected) == -1
            scores = -model.decision_function(selected)
            for position, index in enumerate(indexes):
                model_results[index] = (bool(flags[position]), float(scores[position]))

        typical_income = _typical_monthly_income(rows, kinds, analysis_period)
        expense_rows = rows.loc[kinds.eq("expense")].sort_values("transaction_date", kind="stable")
        predictions: list[dict[str, Any]] = []
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        for index, row in candidates.iterrows():
            model_flag, score = model_results[index]
            assessment = _assess_expense(row, index, expense_rows, typical_income, model_flag, score)
            if assessment is None:
                continue
            severity_counts[assessment["severity"]] += 1
            predictions.append(
                {
                    "transaction_id": str(row["transaction_id"]),
                    "transaction_date": row["transaction_date"].date().isoformat(),
                    "amount": _number(row["amount"]),
                    "category": _clean_category(row["category"]),
                    "merchant": str(row["merchant"]) if pd.notna(row["merchant"]) else None,
                    "description": str(row["description"]) if pd.notna(row["description"]) else None,
                    "anomaly_score": score,
                    "model_flag": model_flag,
                    **assessment,
                }
            )
        severity_rank = {"High": 0, "Medium": 1, "Low": 2}
        predictions.sort(
            key=lambda item: (severity_rank[item["severity"]], -item["risk_points"], -item["anomaly_score"])
        )
        return {
            "status": "available",
            **scope,
            "analyzed_count": int(out_of_sample.sum()),
            "model_flag_count": int(sum(flag for flag, _ in model_results.values())),
            "severity_counts": severity_counts,
            "typical_monthly_income": typical_income,
            "transactions": predictions,
        }

    @staticmethod
    def _run_model_step(label: str, step, *args) -> dict[str, Any]:
        """Run one optional model so its failure cannot break the whole analysis.

        Spending trends, budgets and goals are plain data analysis; an
        incompatible or corrupt saved model should only mark that model as
        unavailable instead of failing insights and recommendations too.
        """
        try:
            return step(*args)
        except Exception as exc:  # noqa: BLE001 - reported as model status
            LOGGER.exception("%s failed; continuing without it.", label)
            return {
                "status": "unavailable",
                "skip_reason": f"{label} failed: {type(exc).__name__}: {exc}",
            }

    def analyze_user(
        self,
        user_id: str,
        analysis_month: str | pd.Period | None = None,
        *,
        include_forecast: bool = True,
        include_anomalies: bool = True,
    ) -> dict[str, Any]:
        requested_user_id = str(user_id).strip()
        if not requested_user_id:
            raise ValueError("User ID cannot be empty.")
        selected_period = self._parse_analysis_month(
            analysis_month if analysis_month is not None else self.analysis_month
        )
        data = load_user_financial_data(self.database_path, requested_user_id)
        transactions = data["transactions"].copy()
        transactions["transaction_date"] = pd.to_datetime(
            transactions["transaction_date"], errors="coerce"
        )
        transactions["amount"] = pd.to_numeric(
            transactions["amount"], errors="coerce"
        )
        invalid_rows = int(
            transactions[["transaction_date", "amount"]].isna().any(axis=1).sum()
        )
        transactions = transactions.dropna(
            subset=["transaction_date", "amount"]
        ).copy()
        if transactions.empty:
            raise ValueError(
                f"User {requested_user_id!r} has no transactions with valid dates and amounts."
            )
        if (transactions["amount"] < 0).any():
            LOGGER.warning(
                "User %s has negative transaction amounts; those rows are retained "
                "as stored and may affect totals.",
                requested_user_id,
            )

        summaries = data["summaries"].copy()
        if not summaries.empty:
            summaries["month"] = summaries["summary_month"].map(_month)
            summaries = summaries.dropna(subset=["month"]).drop_duplicates(
                "month", keep="last"
            ).set_index("month")

        transactions["month"] = transactions["transaction_date"].dt.to_period("M")
        latest_transaction_month = transactions["month"].max()
        available_periods = set(transactions["month"].dropna())
        if not summaries.empty:
            available_periods.update(summaries.index)
        expense_months = transactions.loc[
            transactions["transaction_type"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("expense"),
            "month",
        ]
        latest_expense_month = (
            expense_months.max() if not expense_months.empty else None
        )
        if selected_period is None:
            # Default to the latest month with recorded expenses. A newer month
            # that only holds income (e.g. a salary credited on the 1st) has no
            # spending to analyse, compare or forecast from.
            selected_period = (
                latest_expense_month
                if latest_expense_month is not None
                else latest_transaction_month
            )
        if selected_period not in available_periods:
            raise ValueError(
                f"No transaction or monthly summary data is available for "
                f"{requested_user_id!r} in {selected_period}."
            )
        all_transaction_rows = len(transactions)
        transactions = transactions.loc[transactions["month"] <= selected_period].copy()
        if transactions.empty and (
            summaries.empty or selected_period not in summaries.index
        ):
            raise ValueError(
                f"User {requested_user_id!r} has no transactions through "
                f"{selected_period}."
            )

        category_values, categorization_status = self._predict_missing_categories(
            transactions
        )
        transactions["analysis_category"] = category_values
        transactions["transaction_type_normalized"] = (
            transactions["transaction_type"].astype(str).str.strip().str.casefold()
        )
        transactions["category_for_analysis"] = transactions["analysis_category"]

        latest_month = selected_period
        monthly_transactions = (
            transactions.groupby(
                ["month", "transaction_type_normalized"], as_index=False
            )["amount"]
            .sum()
            .pivot(
                index="month",
                columns="transaction_type_normalized",
                values="amount",
            )
        )
        monthly_transactions = monthly_transactions.rename_axis(None, axis=1)
        monthly_transactions["transaction_income"] = (
            monthly_transactions["income"]
            if "income" in monthly_transactions
            else pd.Series(np.nan, index=monthly_transactions.index)
        )
        monthly_transactions["transaction_expense"] = (
            monthly_transactions["expense"]
            if "expense" in monthly_transactions
            else pd.Series(np.nan, index=monthly_transactions.index)
        )

        monthly_rows: dict[str, dict[str, Any]] = {}
        period_keys = set(monthly_transactions.index)
        if not summaries.empty:
            period_keys.update(
                period for period in summaries.index if period <= selected_period
            )
        for period in sorted(period_keys):
            tx_row = (
                monthly_transactions.loc[period]
                if period in monthly_transactions.index
                else pd.Series(dtype="object")
            )
            summary_row = summaries.loc[period] if period in summaries.index else None
            tx_income = _number(tx_row.get("transaction_income"))
            tx_expense = _number(tx_row.get("transaction_expense"))
            summary_income = (
                _number(summary_row.get("total_income"))
                if summary_row is not None
                else None
            )
            summary_expense = (
                _number(summary_row.get("total_expense"))
                if summary_row is not None
                else None
            )
            expense_rows = transactions.loc[
                (transactions["month"] == period)
                & transactions["transaction_type_normalized"].eq("expense")
            ]
            investment_rows = expense_rows.loc[
                expense_rows["category_for_analysis"].map(
                    self._is_investment_category
                )
            ]
            discretionary_rows = expense_rows.loc[
                expense_rows["category_for_analysis"].map(
                    self._is_discretionary_category
                )
            ]
            monthly_rows[str(period)] = {
                "income": tx_income if tx_income is not None else summary_income,
                "expense": tx_expense if tx_expense is not None else summary_expense,
                "investment_expense": (
                    float(investment_rows["amount"].sum())
                    if not investment_rows.empty
                    else (0.0 if not expense_rows.empty else None)
                ),
                "discretionary_expense": (
                    float(discretionary_rows["amount"].sum())
                    if not expense_rows.empty
                    else None
                ),
                "income_source": "transactions"
                if tx_income is not None
                else ("monthly_financial_summary" if summary_income is not None else None),
                "expense_source": "transactions"
                if tx_expense is not None
                else ("monthly_financial_summary" if summary_expense is not None else None),
            }

        latest_key = str(latest_month)
        latest_totals = monthly_rows.get(latest_key, {})
        latest_income = latest_totals.get("income")
        latest_expense = latest_totals.get("expense")
        net = (
            latest_income - latest_expense
            if latest_income is not None and latest_expense is not None
            else None
        )

        category_rows = transactions.loc[
            (transactions["month"] == latest_month)
            & transactions["category_for_analysis"].notna()
            & transactions["transaction_type_normalized"].eq("expense")
        ]
        category_spending = (
            category_rows.groupby("category_for_analysis")["amount"]
            .sum()
            .sort_values(ascending=False)
        )
        category_trends = []
        # Compare with the most recent (up to three) earlier months that have
        # recorded expenses. A fixed calendar window would leave nothing to compare
        # when recorded history has a gap before the analysis month (for example,
        # a new transaction added today after an older imported dataset).
        prior_expense_rows = transactions.loc[
            (transactions["transaction_type_normalized"] == "expense")
            & (transactions["month"] < latest_month)
            & transactions["category_for_analysis"].notna()
        ]
        observed_prior_months = sorted(prior_expense_rows["month"].unique())[-3:]
        prior_month_window = prior_expense_rows.loc[
            prior_expense_rows["month"].isin(observed_prior_months)
        ]
        for category, current_amount in category_spending.items():
            prior_category_rows = prior_month_window.loc[
                prior_month_window["category_for_analysis"] == category
            ]
            prior_monthly = (
                prior_category_rows.groupby("month")["amount"]
                .sum()
                .reindex(observed_prior_months, fill_value=0.0)
            )
            category_trends.append(
                {
                    "category": str(category),
                    "latest_category_spending": float(current_amount),
                    "historical_months_compared": int(len(prior_monthly)),
                    "historical_average_category_spending": (
                        float(prior_monthly.mean()) if len(prior_monthly) else None
                    ),
                }
            )

        budgets = data["budgets"].copy()
        if not budgets.empty:
            budgets["month"] = budgets["budget_month"].map(_month)
            budgets = budgets[budgets["month"] == latest_month].copy()
        goals = data["goals"].copy()
        profiles = data["users"].copy()
        if include_forecast:
            forecast = self._run_model_step(
                "Expense forecast",
                self._forecast_user_expenses,
                transactions,
                profiles,
                selected_period,
            )
        else:
            forecast = {
                "status": "not_requested",
                "skip_reason": "The expense forecast was not requested for this analysis.",
            }
        if include_anomalies:
            anomaly_results = self._run_model_step(
                "Anomaly detection",
                self._score_potential_anomalies,
                transactions,
                selected_period,
            )
            anomaly_results.setdefault("transactions", [])
        else:
            anomaly_results = {
                "status": "not_requested",
                "skip_reason": "Anomaly detection was not requested for this analysis.",
                "transactions": [],
            }
        goals_out = []
        for row in goals.to_dict(orient="records"):
            goals_out.append(
                {
                    **row,
                    "target_amount": _number(row.get("target_amount")),
                    "current_amount": _number(row.get("current_amount")),
                }
            )

        return {
            "user_id": requested_user_id,
            "analysis_month": latest_key,
            "analysis_period": {
                "selected_month": latest_key,
                "latest_transaction_month": str(latest_transaction_month),
                "latest_expense_month": (
                    str(latest_expense_month)
                    if latest_expense_month is not None
                    else None
                ),
                "latest_available_month": str(max(available_periods)),
                "is_current_month": selected_period
                == pd.Period(self.today, freq="M"),
                "period_status": (
                    "historical"
                    if selected_period < pd.Period(self.today, freq="M")
                    else (
                        "current"
                        if selected_period == pd.Period(self.today, freq="M")
                        else "future"
                    )
                ),
                "data_reaches_current_month": max(available_periods)
                >= pd.Period(self.today, freq="M"),
            },
            "analysis_month_totals": {
                "income": latest_income,
                "expense": latest_expense,
                "net": net,
                "income_source": latest_totals.get("income_source"),
                "expense_source": latest_totals.get("expense_source"),
            },
            "monthly_totals": monthly_rows,
            "category_spending": [
                {"category": str(category), "amount": float(amount)}
                for category, amount in category_spending.items()
            ],
            "category_trends": category_trends,
            "budgets": budgets.drop(columns=["month"], errors="ignore").to_dict(
                orient="records"
            ),
            "goals": goals_out,
            "forecast": forecast,
            "potential_anomalies": anomaly_results,
            "model_status": {
                "categorization": categorization_status,
                "forecast": forecast.get("status"),
                "anomaly_detection": anomaly_results.get("status"),
            },
            "data_quality": {
                "transaction_rows": int(len(transactions)),
                "transactions_after_analysis_month_excluded": int(
                    all_transaction_rows - len(transactions)
                ),
                "invalid_date_or_amount_rows_excluded": invalid_rows,
                "months_with_expense_history": int(
                    transactions.loc[
                        transactions["transaction_type_normalized"].eq("expense"),
                        "month",
                    ].nunique()
                ),
                "uncategorized_transactions": int(
                    transactions["category_for_analysis"].isna().sum()
                ),
                "budget_rows_for_analysis_month": int(len(budgets)),
                "goal_rows": int(len(goals)),
            },
        }

    def _recommendation(
        self,
        user_id: str,
        title: str,
        description: str,
        category: str,
        priority: str,
        reason: str,
        supporting_values: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "recommendation_title": title,
            "recommendation_description": description,
            "recommendation_type": category,
            "priority": priority,
            "reason": reason,
            "supporting_values": supporting_values,
        }

    @staticmethod
    def _deduplicate_recommendations(
        recommendations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        priority_rank = {"High": 0, "Medium": 1, "Low": 2}
        unique: dict[tuple[Any, ...], dict[str, Any]] = {}
        for recommendation in recommendations:
            values = recommendation["supporting_values"]
            transaction_ids = tuple(
                sorted(
                    str(item.get("transaction_id"))
                    for item in values.get("transactions", [])
                    if item.get("transaction_id") is not None
                )
            )
            key = (
                recommendation["user_id"],
                recommendation["recommendation_type"],
                values.get("month") or values.get("forecast_month"),
                values.get("category"),
                values.get("goal_id"),
                transaction_ids,
                recommendation["recommendation_title"]
                if not any(
                    (
                        values.get("month"),
                        values.get("forecast_month"),
                        values.get("category"),
                        values.get("goal_id"),
                        transaction_ids,
                    )
                )
                else None,
            )
            existing = unique.get(key)
            if existing is None or priority_rank.get(
                recommendation["priority"], 3
            ) < priority_rank.get(existing["priority"], 3):
                unique[key] = recommendation
        return list(unique.values())

    def generate_recommendations(self, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        user_id = analysis["user_id"]
        month = analysis["analysis_month"]
        recommendations: list[dict[str, Any]] = []
        totals = analysis["analysis_month_totals"]
        income = _number(totals.get("income"))
        expense = _number(totals.get("expense"))
        net = _number(totals.get("net"))

        if income is not None and income > 0 and expense is not None and expense > income:
            deficit = expense - income
            severity = deficit / income
            priority = "High" if severity >= 0.10 else "Medium"
            monthly_values = analysis["monthly_totals"].get(month, {})
            recommendations.append(
                self._recommendation(
                    user_id,
                    f"Review spending above income for {month}",
                    f"You spent more than you earned in {month}. Look at non-essential "
                    "spending and upcoming bills first; this is not a suggestion to reduce "
                    "investments.",
                    "overspending",
                    priority,
                    "This month’s spending is higher than its income.",
                    {
                        "month": month,
                        "income": income,
                        "expense": expense,
                        "deficit": deficit,
                        "expense_to_income_ratio": expense / income,
                        "investment_expense": _number(
                            monthly_values.get("investment_expense")
                        ),
                        "discretionary_expense": _number(
                            monthly_values.get("discretionary_expense")
                        ),
                    },
                )
            )

        budgets = analysis["budgets"]
        for budget in budgets:
            category = _clean_category(budget.get("category"))
            budget_amount = _number(budget.get("budget_amount"))
            actual_amount = _number(budget.get("actual_amount"))
            if category is None or budget_amount is None or actual_amount is None:
                continue
            if budget_amount <= 0:
                LOGGER.warning(
                    "Skipping budget recommendation for user %s, category %s: "
                    "budget amount is not positive.",
                    user_id,
                    category,
                )
                continue
            utilization = actual_amount / budget_amount
            exceeded_value = budget.get("exceeded")
            exceeded = str(exceeded_value).strip().casefold() in {
                "1",
                "true",
                "yes",
            }
            if actual_amount > budget_amount or exceeded:
                overage = max(actual_amount - budget_amount, 0.0)
                investment_budget = self._is_investment_category(category)
                budget_description = (
                    f"{category} contributions are above the budget you set. Check "
                    "them against your plan; this is not a suggestion to reduce "
                    "investments."
                    if investment_budget
                    else (
                        f"{category} spending is above its budget. Hold back on "
                        f"planned {category.lower()} purchases for the rest of the month."
                        if self._is_discretionary_category(category)
                        else f"{category} spending is above its budget. Check upcoming "
                        f"{category.lower()} payments against your plan."
                    )
                )
                recommendations.append(
                    self._recommendation(
                        user_id,
                        f"{category} budget exceeded in {month}",
                        budget_description,
                        "budget_warning",
                        "High",
                        "Spending in this category is higher than its budget.",
                        {
                            "month": month,
                            "category": category,
                            "budget_amount": budget_amount,
                            "actual_amount": actual_amount,
                            "over_budget_amount": overage,
                            "utilization": utilization,
                        },
                    )
                )
            elif utilization >= self.budget_warning_threshold:
                budget_description = (
                    f"Investment allocations have used {utilization:.0%} of the "
                    "budget. Check planned contributions before changing them."
                    if self._is_investment_category(category)
                    else f"{category} spending has used {utilization:.0%} of its "
                    "budget. Check what is still planned before spending more."
                )
                recommendations.append(
                    self._recommendation(
                        user_id,
                        f"Monitor {category} budget for {month}",
                        budget_description,
                        "budget_warning",
                        "Medium",
                        f"Spending has reached {self.budget_warning_threshold:.0%} "
                        "or more of the budget.",
                        {
                            "month": month,
                            "category": category,
                            "budget_amount": budget_amount,
                            "actual_amount": actual_amount,
                            "remaining_budget": budget_amount - actual_amount,
                            "utilization": utilization,
                        },
                    )
                )

        budget_alert_categories = {
            str(item["supporting_values"].get("category", "")).casefold()
            for item in recommendations
            if item["recommendation_type"] == "budget_warning"
        }
        for trend in analysis["category_trends"]:
            category = trend["category"]
            if (
                not self._is_discretionary_category(category)
                or self._is_investment_category(category)
                or str(category).casefold() in budget_alert_categories
            ):
                continue
            current_amount = trend["latest_category_spending"]
            prior_average = trend["historical_average_category_spending"]
            prior_month_count = trend["historical_months_compared"]
            if prior_month_count < 2 or prior_average is None:
                continue
            category_increase = float(current_amount) - float(prior_average)
            if (
                prior_average > 0
                and category_increase > prior_average * self.category_increase_threshold
            ):
                recommendations.append(
                    self._recommendation(
                        user_id,
                        f"Review {category} spending increase",
                        f"{category} spending in {month} is above its average for "
                        "available prior months. Consider which purchases can be reduced.",
                        "expense_reduction",
                        "Medium",
                        "This category’s latest monthly spending exceeds its available "
                        "prior-three-month average by the configured trend threshold.",
                        {
                            "month": month,
                            "category": category,
                            "latest_category_spending": float(current_amount),
                            "historical_months_compared": prior_month_count,
                            "historical_average_category_spending": float(prior_average),
                            "increase_amount": category_increase,
                            "increase_rate": category_increase / prior_average,
                        },
                    )
                )

        monthly_totals = analysis["monthly_totals"]
        current_period = pd.Period(month, freq="M")
        latest_discretionary = _number(
            monthly_totals.get(month, {}).get("discretionary_expense")
        )
        historical_discretionary: list[float] = []
        for offset in range(1, 4):
            prior_month = str(current_period - offset)
            prior = monthly_totals.get(prior_month)
            prior_amount = (
                _number(prior.get("discretionary_expense")) if prior else None
            )
            if prior_amount is not None:
                historical_discretionary.append(prior_amount)
        has_specific_spending_advice = any(
            item["recommendation_type"] in {"expense_reduction", "budget_warning"}
            and item["supporting_values"].get("category") is not None
            for item in recommendations
        )
        if (
            latest_discretionary is not None
            and len(historical_discretionary) >= 2
            and not has_specific_spending_advice
        ):
            historical_average = float(np.mean(historical_discretionary))
            increase = latest_discretionary - historical_average
            if (
                historical_average > 0
                and increase > historical_average * self.category_increase_threshold
            ):
                recommendations.append(
                    self._recommendation(
                        user_id,
                        f"Review non-investment spending increase in {month}",
                        "Spending on non-essentials (shopping, eating out, "
                        "entertainment and similar) is higher than usual. Review those "
                        "purchases; investments are not included.",
                        "expense_reduction",
                        "Medium",
                        "Tracked discretionary-category spending exceeds the average "
                        "of available preceding months in the three-month window.",
                        {
                            "month": month,
                            "latest_discretionary_expense": latest_discretionary,
                            "historical_months_compared": len(
                                historical_discretionary
                            ),
                            "historical_average_discretionary_expense": (
                                historical_average
                            ),
                            "increase_amount": increase,
                            "increase_rate": increase / historical_average,
                            "investment_expense_excluded": _number(
                                monthly_totals.get(month, {}).get(
                                    "investment_expense"
                                )
                            ),
                        },
                    )
                )

        goals = analysis["goals"]
        available_surplus = net if net is not None and net > 0 else None
        if net is None:
            affordability_note = (
                f"Income and expense totals are unavailable for {month}, so "
                "affordability cannot be assessed."
            )
            affordability_reason = (
                "Income or expense data was unavailable for the selected month."
            )
        else:
            affordability_note = (
                f"No positive surplus was recorded in {month}, so affordability "
                "cannot be confirmed."
            )
            affordability_reason = (
                "The analyzed month did not show a positive recorded surplus."
            )
        for goal in goals:
            target = _number(goal.get("target_amount"))
            current = _number(goal.get("current_amount"))
            status = str(goal.get("status") or "").strip().casefold()
            if target is None or current is None or target <= 0 or current >= target:
                continue
            if status not in {"in progress", "active", "ongoing"}:
                continue
            remaining = max(target - current, 0.0)
            target_date = pd.to_datetime(goal.get("target_date"), errors="coerce")
            goal_type = str(goal.get("goal_type") or "financial goal")
            goal_values = {
                "goal_id": goal.get("goal_id"),
                "goal_type": goal_type,
                "target_amount": target,
                "current_amount": current,
                "remaining_amount": remaining,
                "progress_rate": current / target,
                "progress_percentage": current / target * 100,
                "analysis_month": month,
                "target_date": str(goal.get("target_date"))
                if pd.notna(target_date)
                else None,
                "priority": goal.get("priority"),
            }
            if pd.isna(target_date):
                LOGGER.warning(
                    "Skipping schedule-based goal recommendation for user %s: "
                    "goal %s has an invalid target date.",
                    user_id,
                    goal.get("goal_id"),
                )
                continue
            target_period = target_date.to_period("M")
            current_period_for_deadline = pd.Period(month, freq="M")
            if target_period < current_period_for_deadline:
                recommendations.append(
                    self._recommendation(
                        user_id,
                        f"Review overdue {goal_type} goal",
                        f"The target month for this {goal_type} goal is earlier than "
                        f"the analyzed month {month}, while the goal is still marked "
                        "active. Review the remaining amount and update its timeline.",
                        "goal_progress",
                        "High",
                        "The stored target month precedes the selected analysis month; "
                        "the comparison does not use the computer's current date.",
                        goal_values,
                    )
                )
                continue

            months_left = max(
                1,
                (target_period.year - current_period_for_deadline.year) * 12
                + target_period.month
                - current_period_for_deadline.month
                + 1,
            )
            required_monthly = remaining / months_left
            goal_values["months_remaining_including_current"] = months_left
            goal_values["required_monthly_contribution"] = required_monthly
            if available_surplus is not None:
                if available_surplus >= required_monthly:
                    recommendations.append(
                        self._recommendation(
                            user_id,
                            f"Plan contributions toward {goal_type}",
                            f"The recorded surplus for {month} could cover the "
                            f"calculated monthly contribution of {required_monthly:.2f} "
                            "needed to reach this goal by its stored target date.",
                            "goal_progress",
                            "Medium",
                            "The goal target month is not before the analysis month, and "
                            "the analyzed month's observed "
                            "surplus is at least the calculated monthly requirement.",
                            {
                                **goal_values,
                                "latest_monthly_surplus": available_surplus,
                            },
                        )
                    )
                else:
                    recommendations.append(
                        self._recommendation(
                            user_id,
                            f"Reassess timeline for {goal_type}",
                            "The calculated monthly contribution needed for the stored "
                            "goal date is higher than the analyzed month's recorded "
                            "surplus. "
                            "Review the goal amount, date, or contribution plan.",
                            "goal_progress",
                            "High"
                            if str(goal.get("priority") or "").casefold() == "high"
                            else "Medium",
                            "This comparison uses the analyzed month's observed surplus "
                            "and the remaining goal amount divided by calendar months from the "
                            "analysis month through the target month.",
                            {
                                **goal_values,
                                "latest_monthly_surplus": available_surplus,
                            },
                        )
                    )
            else:
                recommendations.append(
                    self._recommendation(
                        user_id,
                        f"Review contribution plan for {goal_type}",
                        f"The remaining goal amount is {remaining:.2f}; reaching it by "
                        f"{goal_values['target_date']} would require about "
                        f"{required_monthly:.2f} per month. {affordability_note}",
                        "goal_progress",
                        "High"
                        if str(goal.get("priority") or "").casefold() == "high"
                        else "Medium",
                        "The contribution estimate uses the stored goal balance and "
                        "calendar months from the selected analysis period. "
                        + affordability_reason,
                        goal_values,
                    )
                )

        if available_surplus is not None and not any(
            recommendation["recommendation_type"] == "goal_progress"
            for recommendation in recommendations
        ):
            recommendations.append(
                self._recommendation(
                    user_id,
                    f"Review available surplus from {month}",
                    f"You earned more than you spent in {month}. Consider moving "
                    "some of the surplus into savings.",
                    "saving_opportunity",
                    "Low",
                    "Based on the difference between this month’s income and "
                    "spending.",
                    {
                        "month": month,
                        "monthly_surplus": available_surplus,
                        "income": income,
                        "expense": expense,
                    },
                )
            )

        forecast = analysis["forecast"]
        if forecast.get("status") == "available":
            forecast_month = _month(forecast.get("forecast_month"))
            predicted = _number(forecast.get("predicted_expense"))
            if forecast_month is not None and predicted is not None:
                current_calendar_month = pd.Period(self.today, freq="M")
                if forecast_month < current_calendar_month:
                    forecast["status"] = "stale"
                    forecast["skip_reason"] = (
                        f"Forecast month {forecast_month} is not later than the "
                        f"current month {current_calendar_month}."
                    )
                    LOGGER.info(
                        "Omitting stale forecast for user %s: forecast month %s.",
                        user_id,
                        forecast_month,
                    )
                else:
                    historical = [
                        _number(values.get("expense"))
                        for key, values in monthly_totals.items()
                        if _month(key) is not None
                        and _month(key) < forecast_month
                        and _number(values.get("expense")) is not None
                    ]
                    recent_average = (
                        float(np.mean(historical[-3:])) if historical else None
                    )
                    income_reference = (
                        _number(forecast.get("monthly_income_profile"))
                        or income
                    )
                    exceeds_income = (
                        income_reference is not None
                        and income_reference > 0
                        and predicted > income_reference
                    )
                    exceeds_trend = (
                        recent_average is not None
                        and recent_average > 0
                        and predicted > recent_average * (1 + self.category_increase_threshold)
                    )
                    if exceeds_income or exceeds_trend:
                        reason_parts = []
                        if exceeds_income:
                            reason_parts.append("the forecast exceeds profile monthly income")
                        if exceeds_trend:
                            reason_parts.append(
                                "the forecast is above the recent recorded expense average"
                            )
                        recommendations.append(
                            self._recommendation(
                                user_id,
                                f"Plan for forecasted expenses in {forecast_month}",
                                "The existing expense forecast is elevated against "
                                "available income or recent spending. Review upcoming "
                                "commitments and budget allocations.",
                                "future_expense_warning",
                                "High" if exceeds_income else "Medium",
                                "The forecast " + " and ".join(reason_parts) + ".",
                                {
                                    "forecast_month": str(forecast_month),
                                    "predicted_expense": predicted,
                                    "monthly_income_reference": income_reference,
                                    "recent_average_expense": recent_average,
                                    "forecast_model_history_months": forecast.get(
                                        "history_months"
                                    ),
                                },
                            )
                        )

        anomaly_data = analysis["potential_anomalies"]
        if anomaly_data.get("status") == "available":
            current_month_candidates = [
                item
                for item in anomaly_data["transactions"]
                if _month(item["transaction_date"]) == current_period
            ]
            if current_month_candidates:
                severities = [item.get("severity", "Low") for item in current_month_candidates]
                high = severities.count("High")
                medium = severities.count("Medium")
                priority = "High" if high else "Medium" if medium else "Low"
                important = high + medium
                recommendations.append(
                    self._recommendation(
                        user_id,
                        f"Review unusual transactions from {month}",
                        (
                            f"{important} transaction(s) in {month} stand out from your usual "
                            f"spending ({high} high, {medium} medium severity)"
                            if important
                            else f"{len(current_month_candidates)} transaction(s) in {month} follow an unusual pattern"
                        )
                        + ". Check them on the Anomaly detection page; they are not "
                        "confirmed fraud.",
                        "unusual_transaction_review",
                        priority,
                        "Flags combine the saved behavioral Isolation Forest with checks "
                        "against your own earlier spending: amount versus your usual "
                        "purchase in the category, first payments to a merchant, share of "
                        "monthly income and repeated identical charges. Stored anomaly "
                        "labels are not used.",
                        {
                            "month": month,
                            "model_training_end": anomaly_data.get("training_end"),
                            "flagged_transaction_count": len(current_month_candidates),
                            "high_severity_count": high,
                            "medium_severity_count": medium,
                            "transactions": [
                                {
                                    key: item[key]
                                    for key in (
                                        "transaction_id", "transaction_date", "amount",
                                        "category", "merchant", "anomaly_score", "severity",
                                    )
                                    if key in item
                                }
                                for item in current_month_candidates
                            ],
                            "label_provenance": "unverified",
                        },
                    )
                )

        recommendations = self._deduplicate_recommendations(recommendations)
        return sorted(
            recommendations,
            key=lambda item: (
                {"High": 0, "Medium": 1, "Low": 2}.get(item["priority"], 3),
                item["recommendation_type"],
                item["recommendation_title"],
            ),
        )

    def generate(
        self,
        user_id: str,
        analysis_month: str | pd.Period | None = None,
    ) -> dict[str, Any]:
        analysis = self.analyze_user(user_id, analysis_month=analysis_month)
        recommendations = self.generate_recommendations(analysis)
        forecast = analysis["forecast"]
        recommendation_skips = []
        if forecast.get("status") != "available":
            recommendation_skips.append(
                {
                    "recommendation_type": "future_expense_warning",
                    "reason": forecast.get("skip_reason")
                    or (
                        "Forecast-based recommendation skipped because forecast "
                        f"status is {forecast.get('status', 'unknown')}."
                    ),
                }
            )
        return {
            "user_id": analysis["user_id"],
            "analysis_month": analysis["analysis_month"],
            "analysis_period": analysis["analysis_period"],
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
            "recommendation_skips": recommendation_skips,
            "model_status": analysis["model_status"],
            "data_quality": analysis["data_quality"],
            "label_provenance": "unverified",
        }

    def evaluate_all_users(self) -> dict[str, Any]:
        path = Path(self.database_path).expanduser()
        resolved = (path if path.is_absolute() else Path.cwd() / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Finance database not found: {resolved}")
        connection = sqlite3.connect(
            f"file:{resolved.as_posix()}?mode=ro",
            uri=True,
        )
        try:
            user_ids = [
                str(row[0])
                for row in connection.execute(
                    'SELECT DISTINCT "user_id" FROM "transactions" ORDER BY "user_id"'
                )
            ]
        except sqlite3.Error as exc:
            raise sqlite3.DatabaseError(
                f"Unable to list users for recommendation evaluation: {exc}"
            ) from exc
        finally:
            connection.close()

        category_counts: Counter[str] = Counter()
        priority_counts: Counter[str] = Counter()
        error_count = 0
        users_with_recommendations = 0
        model_status_counts: dict[str, Counter[str]] = {
            key: Counter() for key in ("categorization", "forecast", "anomaly_detection")
        }
        for user_id in user_ids:
            try:
                result = self.generate(user_id)
            except (ValueError, sqlite3.Error, OSError) as exc:
                LOGGER.error(
                    "Recommendation evaluation failed for user %s: %s", user_id, exc
                )
                error_count += 1
                continue
            if result["recommendations"]:
                users_with_recommendations += 1
            category_counts.update(
                item["recommendation_type"]
                for item in result["recommendations"]
            )
            priority_counts.update(
                item["priority"] for item in result["recommendations"]
            )
            for name, status in result["model_status"].items():
                if isinstance(status, dict):
                    status = status.get("status", "unknown")
                model_status_counts[name][str(status)] += 1

        return {
            "users_evaluated": len(user_ids),
            "users_with_recommendations": users_with_recommendations,
            "recommendation_count": sum(category_counts.values()),
            "recommendations_by_type": dict(category_counts),
            "recommendations_by_priority": dict(priority_counts),
            "model_status_by_user": {
                name: dict(counts) for name, counts in model_status_counts.items()
            },
            "user_errors": error_count,
            "interpretation": (
                "Operational coverage summary only; no verified recommendation "
                "outcome labels are available to measure effectiveness."
            ),
        }
