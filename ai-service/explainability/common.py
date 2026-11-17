from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


EXPLAINABILITY_DIR = Path(__file__).resolve().parent
AI_SERVICE_DIR = EXPLAINABILITY_DIR.parent
PROJECT_ROOT = AI_SERVICE_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "explainability"


def load_source_module(module_name: str, path: Path) -> ModuleType:
    """Load a sibling script from the hyphenated service directory."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def require_shap():
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explanations. Install the project requirements "
            "that include the pinned shap dependency."
        ) from exc
    return shap


def output_directory(output_dir: str | Path | None = None) -> Path:
    target = Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_filename(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return result or "explanation"


def deterministic_sample(
    frame: pd.DataFrame,
    max_rows: int,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    if max_rows < 1:
        raise ValueError("max_rows must be greater than zero.")
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=random_state).sort_index()


def single_output_explanation(explanation: Any, output_index: int = 0):
    """Select one output from SHAP's single- or multi-output Explanation."""
    shap = require_shap()
    values = np.asarray(explanation.values)
    if values.ndim == 2:
        if output_index != 0:
            raise IndexError("The explanation contains only one output.")
        return explanation
    if values.ndim != 3:
        raise ValueError(
            f"Unsupported SHAP value dimensions: expected 2 or 3, got {values.ndim}."
        )
    if not 0 <= output_index < values.shape[2]:
        raise IndexError(f"Output index is out of range: {output_index}.")

    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 0:
        selected_base = base_values
    elif base_values.ndim == 1:
        selected_base = (
            base_values
            if len(base_values) == values.shape[0]
            else base_values[output_index : output_index + 1]
        )
    else:
        selected_base = base_values[:, output_index]

    return shap.Explanation(
        values=values[:, :, output_index],
        base_values=selected_base,
        data=explanation.data,
        feature_names=explanation.feature_names,
    )


def json_number(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def save_shap_plots(
    explanation: Any,
    output_stem: Path,
    *,
    max_display: int = 20,
    title: str | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap = require_shap()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for chart, plot_function in (
        ("bar", shap.plots.bar),
        ("beeswarm", shap.plots.beeswarm),
    ):
        plot_function(explanation, max_display=max_display, show=False)
        if title:
            plt.gca().set_title(title)
        destination = output_stem.with_name(f"{output_stem.name}_{chart}.png")
        plt.tight_layout()
        plt.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close()
        generated.append(destination)
    return generated


def save_waterfall(
    explanation: Any,
    output_path: Path,
    *,
    max_display: int = 20,
    title: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap = require_shap()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    if title:
        plt.gca().set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path


def explanation_rows(explanation: Any, row_index: int = 0) -> list[dict[str, Any]]:
    values = np.asarray(explanation.values)[row_index]
    data = np.asarray(explanation.data, dtype=object)[row_index]
    names = list(explanation.feature_names)
    return [
        {
            "feature": str(name),
            "value": _json_value(value),
            "shap_value": json_number(shap_value),
        }
        for name, value, shap_value in zip(names, data, values)
    ]


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_global_importance(
    path: Path,
    feature_names: list[str],
    mean_absolute_values: np.ndarray,
) -> Path:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.asarray(mean_absolute_values, dtype=float),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    return path
