"""Predictive modeling training script.

This module loads parquet data, engineers features defined in a YAML
configuration, trains baseline gradient boosting models, and logs
metrics/artefacts to a local MLflow tracking directory.

The functions that prepare the supervised dataset are shared with the
reinforcement learning scaffolding under ``models.rl`` to guarantee
consistent data semantics across modeling approaches.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_ARTIFACT_DIR = Path("artifacts")


@dataclass(frozen=True)
class DatasetBundle:
    """Container returned by :func:`prepare_supervised_dataset`."""

    frame: pl.DataFrame
    feature_columns: Sequence[str]
    target_column: str
    timestamp_column: str
    symbol_column: str | None = None


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the YAML configuration file."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, MutableMapping):
        raise ValueError("Configuration file must define a mapping at the top level.")
    return dict(config)


def load_parquet_dataset(data_path: Path) -> pl.DataFrame:
    """Load a parquet dataset into a Polars DataFrame."""

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    frame = pl.read_parquet(str(data_path))
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("Expected a Polars DataFrame from parquet reader")
    return frame


def _normalize_inherits(entry: Mapping[str, Any]) -> List[str]:
    inherits = entry.get("inherits", [])
    if isinstance(inherits, str):
        return [inherits]
    if inherits is None:
        return []
    if not isinstance(inherits, Iterable):
        raise TypeError("'inherits' must be a string or iterable of strings")
    return list(inherits)


def resolve_feature_set(feature_sets: Mapping[str, Mapping[str, Any]], name: str) -> List[Mapping[str, Any]]:
    """Resolve a feature set definition, expanding inherited feature sets."""

    if name not in feature_sets:
        available = ", ".join(sorted(feature_sets))
        raise KeyError(f"Unknown feature set '{name}'. Available options: {available}")

    entry = feature_sets[name]
    if not isinstance(entry, Mapping):
        raise TypeError(f"Feature set '{name}' must be a mapping definition")

    features: List[Mapping[str, Any]] = []
    for parent in _normalize_inherits(entry):
        features.extend(resolve_feature_set(feature_sets, parent))

    own_features = entry.get("features", [])
    if not isinstance(own_features, Iterable):
        raise TypeError(f"Feature set '{name}' must define an iterable under 'features'")

    for feature in own_features:
        if not isinstance(feature, Mapping):
            raise TypeError("Each feature definition must be a mapping")
        features.append(feature)

    return features


def _feature_alias(definition: Mapping[str, Any]) -> str:
    source = definition.get("source")
    kind = definition.get("kind")
    rename = definition.get("rename")
    if rename:
        return str(rename)
    if source and kind:
        return f"{source}_{kind}"
    raise ValueError("Feature definitions must include either a 'rename' or both 'source' and 'kind'.")


def _feature_expression(definition: Mapping[str, Any]) -> pl.Expr:
    """Translate a feature definition into a Polars expression."""

    kind = definition.get("kind")
    source = definition.get("source")
    if not kind or not source:
        raise ValueError("Feature definitions must include 'kind' and 'source'.")

    column = pl.col(str(source))
    rename = _feature_alias(definition)

    if kind == "pct_change":
        periods = int(definition.get("periods", 1))
        expr = column.pct_change(periods)
    elif kind == "rolling_mean":
        window = int(definition["window"])
        min_periods = int(definition.get("min_periods", window))
        expr = column.rolling_mean(window_size=window, min_periods=min_periods)
    elif kind == "rolling_std":
        window = int(definition["window"])
        min_periods = int(definition.get("min_periods", window))
        expr = column.rolling_std(window_size=window, min_periods=min_periods)
    elif kind == "rolling_zscore":
        window = int(definition["window"])
        min_periods = int(definition.get("min_periods", window))
        mean_expr = column.rolling_mean(window_size=window, min_periods=min_periods)
        std_expr = column.rolling_std(window_size=window, min_periods=min_periods)
        expr = ((column - mean_expr) / std_expr).fill_nan(0.0).fill_null(0.0)
    elif kind == "ema":
        span = float(definition["span"])
        alpha = 2.0 / (span + 1.0)
        expr = column.ewm_mean(alpha=alpha, adjust=True)
    elif kind == "lag":
        periods = int(definition.get("periods", 1))
        expr = column.shift(periods)
    else:
        raise ValueError(f"Unsupported feature kind: {kind}")

    return expr.alias(rename)


def engineer_features(frame: pl.DataFrame, feature_definitions: Sequence[Mapping[str, Any]]) -> Tuple[pl.DataFrame, List[str]]:
    """Add engineered features to the frame based on configuration definitions."""

    expressions: List[pl.Expr] = []
    feature_names: List[str] = []
    for definition in feature_definitions:
        expressions.append(_feature_expression(definition))
        feature_names.append(_feature_alias(definition))
    if expressions:
        frame = frame.with_columns(expressions)
    return frame, feature_names


def append_target(frame: pl.DataFrame, target_cfg: Mapping[str, Any]) -> Tuple[pl.DataFrame, str]:
    """Create the supervised learning target column."""

    if not isinstance(target_cfg, Mapping):
        raise TypeError("Target configuration must be a mapping")

    name = str(target_cfg.get("name", "target"))
    kind = target_cfg.get("kind", "future_return")
    source = str(target_cfg.get("source", "close"))
    horizon = int(target_cfg.get("horizon", 1))
    column = pl.col(source)

    if kind == "future_return":
        shifted = column.shift(-horizon)
        expr = ((shifted / column) - 1.0).alias(name)
    elif kind == "future_value":
        expr = column.shift(-horizon).alias(name)
    else:
        raise ValueError(f"Unsupported target kind: {kind}")

    frame = frame.with_columns(expr)
    return frame, name


def prepare_supervised_dataset(
    config: Mapping[str, Any],
    data_path: Path,
    feature_set: str,
) -> DatasetBundle:
    """Load parquet data and engineer features for supervised modeling."""

    frame = load_parquet_dataset(data_path)
    data_cfg = config.get("data", {})
    timestamp_column = str(data_cfg.get("timestamp_column", "timestamp"))
    symbol_column = data_cfg.get("symbol_column")

    if timestamp_column not in frame.columns:
        raise KeyError(f"Timestamp column '{timestamp_column}' not found in dataset")

    frame = frame.sort(timestamp_column)

    feature_sets = config.get("feature_sets", {})
    feature_definitions = resolve_feature_set(feature_sets, feature_set)
    frame, feature_columns = engineer_features(frame, feature_definitions)

    target_cfg = data_cfg.get("target")
    if not target_cfg:
        raise KeyError("Configuration must specify data.target")
    frame, target_column = append_target(frame, target_cfg)

    required_columns = list(feature_columns) + [target_column]
    frame = frame.drop_nulls(subset=required_columns)

    return DatasetBundle(
        frame=frame,
        feature_columns=feature_columns,
        target_column=target_column,
        timestamp_column=timestamp_column,
        symbol_column=str(symbol_column) if symbol_column else None,
    )


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series, utc=False, errors="coerce")


def _train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def _evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    directional_hits = np.sign(predictions) == np.sign(y_true)
    directional_accuracy = float(np.mean(directional_hits))
    strategy_returns = np.sign(predictions) * y_true
    average_return = float(np.mean(strategy_returns))
    volatility = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0.0
    sharpe = float(average_return / volatility) if volatility > 1e-12 else 0.0
    return {
        "rmse": rmse,
        "mae": float(mae),
        "mse": float(mse),
        "r2": float(r2),
        "directional_accuracy": directional_accuracy,
        "avg_strategy_return": average_return,
        "strategy_volatility": volatility,
        "strategy_sharpe": sharpe,
    }


def _log_metrics(metrics: Mapping[str, float]) -> None:
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def _save_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def run_single_split(
    dataset: DatasetBundle,
    pandas_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    split_name: str,
    split_cfg: Mapping[str, Any],
    artifact_dir: Path,
    feature_set: str,
) -> Dict[str, float]:
    """Train and evaluate a model for a single train/validation split."""

    timestamps = _ensure_datetime(pandas_frame[dataset.timestamp_column])
    train_start = pd.Timestamp(split_cfg["train_start"])
    train_end = pd.Timestamp(split_cfg["train_end"])
    validation_start = pd.Timestamp(split_cfg["validation_start"])
    validation_end = pd.Timestamp(split_cfg["validation_end"])

    train_mask = (timestamps >= train_start) & (timestamps <= train_end)
    val_mask = (timestamps >= validation_start) & (timestamps <= validation_end)

    X_train = pandas_frame.loc[train_mask, feature_columns]
    y_train = pandas_frame.loc[train_mask, target_column]
    X_val = pandas_frame.loc[val_mask, feature_columns]
    y_val = pandas_frame.loc[val_mask, target_column]

    if X_train.empty or X_val.empty:
        raise ValueError(
            f"Split '{split_name}' results in empty train or validation set. Check split boundaries."
        )

    model = _train_model(X_train, y_train)
    predictions = model.predict(X_val)
    metrics = _evaluate_predictions(y_val, predictions)

    mlflow.log_params(
        {
            "model": "GradientBoostingRegressor",
            "feature_set": feature_set,
            "split": split_name,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
        }
    )
    _log_metrics(metrics)

    feature_importances = dict(zip(feature_columns, model.feature_importances_.tolist()))
    mlflow.log_dict(feature_importances, "feature_importances.json")

    model_dir = artifact_dir / "models" / feature_set
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / f"{split_name}_metrics.json"
    _save_json(metrics, metrics_path)

    import joblib

    model_path = model_dir / f"{split_name}_gradient_boosting.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    return metrics


def generate_rolling_windows(
    timestamps: pd.Series,
    window_cfg: Mapping[str, Any],
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yield rolling window boundaries (train_start, train_end, val_start, val_end)."""

    training_days = int(window_cfg["training_days"])
    validation_days = int(window_cfg["validation_days"])
    step_days = int(window_cfg.get("step_days", validation_days))
    min_train_date = pd.Timestamp(window_cfg.get("min_train_date", timestamps.min()))

    start = max(min_train_date, timestamps.min())
    end_limit = timestamps.max()

    while True:
        train_start = start
        train_end = train_start + pd.Timedelta(days=training_days - 1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=validation_days - 1)
        if val_end > end_limit:
            break
        yield train_start, train_end, val_start, val_end
        start = start + pd.Timedelta(days=step_days)


def run_rolling_cross_validation(
    dataset: DatasetBundle,
    pandas_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    cv_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    """Execute rolling-window cross-validation and aggregate metrics."""

    results: Dict[str, float] = {}
    timestamps = _ensure_datetime(pandas_frame[dataset.timestamp_column])

    for window_cfg in cv_cfg.get("rolling_windows", []):
        window_name = window_cfg.get("name", "rolling_window")
        fold_metrics: List[Dict[str, float]] = []
        for train_start, train_end, val_start, val_end in generate_rolling_windows(timestamps, window_cfg):
            train_mask = (timestamps >= train_start) & (timestamps <= train_end)
            val_mask = (timestamps >= val_start) & (timestamps <= val_end)
            X_train = pandas_frame.loc[train_mask, feature_columns]
            y_train = pandas_frame.loc[train_mask, target_column]
            X_val = pandas_frame.loc[val_mask, feature_columns]
            y_val = pandas_frame.loc[val_mask, target_column]
            if X_train.empty or X_val.empty:
                continue
            model = _train_model(X_train, y_train)
            predictions = model.predict(X_val)
            fold_metrics.append(_evaluate_predictions(y_val, predictions))

        if not fold_metrics:
            logger.warning("No valid folds generated for rolling window '%s'", window_name)
            continue

        aggregated = {
            f"cv_{window_name}_{metric}": float(np.mean([m[metric] for m in fold_metrics]))
            for metric in fold_metrics[0]
        }
        results.update(aggregated)

    for key, value in results.items():
        mlflow.log_metric(key, value)
    return results


def configure_mlflow(artifact_dir: Path, experiment_name: str) -> None:
    tracking_path = artifact_dir / "mlruns"
    tracking_uri = f"file://{tracking_path.resolve()}"
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline predictive models")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    parser.add_argument("--data-path", type=Path, help="Parquet dataset path. Overrides config.data.path", required=False)
    parser.add_argument("--feature-set", type=str, default="basic_price", help="Feature set key to train")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="Directory for stored artefacts")
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    data_path = args.data_path or Path(data_cfg.get("path", ""))
    if not data_path:
        raise ValueError("Data path must be supplied via --data-path or config.data.path")

    dataset = prepare_supervised_dataset(config, data_path, args.feature_set)
    pandas_frame = dataset.frame.to_pandas()
    pandas_frame[dataset.timestamp_column] = _ensure_datetime(pandas_frame[dataset.timestamp_column])

    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = args.experiment or config.get("mlflow", {}).get("experiment", "predictive_baselines")
    configure_mlflow(artifact_dir, experiment_name)

    splits = config.get("splits", {})
    if not splits:
        raise ValueError("No train/validation splits defined in configuration")

    for split_name, split_cfg in splits.items():
        run_name = f"{args.feature_set}__{split_name}"
        with mlflow.start_run(run_name=run_name):
            logger.info("Training feature set '%s' on split '%s'", args.feature_set, split_name)
            metrics = run_single_split(
                dataset=dataset,
                pandas_frame=pandas_frame,
                feature_columns=dataset.feature_columns,
                target_column=dataset.target_column,
                split_name=split_name,
                split_cfg=split_cfg,
                artifact_dir=artifact_dir,
                feature_set=args.feature_set,
            )
            logger.info("Validation metrics for %s: %s", split_name, metrics)
            cv_cfg = config.get("cross_validation", {})
            if cv_cfg:
                cv_metrics = run_rolling_cross_validation(
                    dataset=dataset,
                    pandas_frame=pandas_frame,
                    feature_columns=dataset.feature_columns,
                    target_column=dataset.target_column,
                    cv_cfg=cv_cfg,
                )
                logger.info("Cross-validation metrics: %s", cv_metrics)


if __name__ == "__main__":
    main()
