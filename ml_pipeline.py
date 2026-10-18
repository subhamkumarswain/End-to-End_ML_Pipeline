"""Train and use a tabular classification pipeline from CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _read_csv(csv_path: Path) -> pd.DataFrame:
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), None)
    if not header:
        raise ValueError(f"CSV must include a header row: {csv_path}")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError(f"CSV contains an empty column name: {csv_path}")
        if column in seen:
            duplicates.add(column)
        seen.add(column)
    if duplicates:
        raise ValueError(f"CSV contains duplicate column names: {sorted(duplicates)}")
    return pd.read_csv(csv_path)


def _build_model(frame: pd.DataFrame) -> Pipeline:
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in frame.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def _validate_features(frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty:
        raise ValueError("Input data must contain at least one row")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing required feature columns: {missing}")
    empty_columns = [column for column in columns if frame[column].isna().all()]
    if empty_columns:
        raise ValueError(
            f"Feature columns contain only missing values: {empty_columns}"
        )
    infinite_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].isin([math.inf, -math.inf]).any()
    ]
    if infinite_columns:
        raise ValueError(
            f"Numeric features contain infinite values: {infinite_columns}"
        )


def _create_model_version(prefix: str = "model") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _persist_model_artifact(
    model_path: Path,
    bundle: dict[str, Any],
    version: str | None = None,
) -> tuple[Path, str | None]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    resolved_version = version
    if version is not None:
        versioned_path = model_path.with_name(
            f"{model_path.stem}-{version}{model_path.suffix}"
        )
        versioned_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, versioned_path)
        resolved_version = version
    return model_path, resolved_version


def predict_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Validate feature columns and return a copy with predictions added."""
    if not isinstance(bundle, dict) or not {
        "model",
        "features",
        "target",
    }.issubset(bundle):
        raise ValueError("Model artifact has an invalid format")
    features = bundle["features"]
    if not isinstance(features, list) or not all(
        isinstance(column, str) for column in features
    ):
        raise ValueError("Model artifact has an invalid feature schema")
    _validate_features(frame, features)

    result = frame.copy()
    prediction_column = "prediction"
    if prediction_column in result.columns:
        raise ValueError(f"Input data already contains a '{prediction_column}' column")
    result[prediction_column] = bundle["model"].predict(result[features])
    return result


def _evaluate_model(
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = pd.unique(y_test).tolist()
    precision_macro = float(
        precision_score(y_test, predictions, average="macro", zero_division=0)
    )
    recall_macro = float(
        recall_score(y_test, predictions, average="macro", zero_division=0)
    )
    f1_macro = float(f1_score(y_test, predictions, average="macro", zero_division=0))
    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": labels,
            "values": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        },
    }


def _log_mlflow_run(
    experiment_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    artifact_paths: list[Path],
) -> None:
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking requires the optional dependencies. "
            "Install them with: python -m pip install -r requirements-mlflow.txt"
        ) from error

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else str(value)
                for key, value in parameters.items()
                if value is not None
            }
        )
        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if value is not None}
        )
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))


def monitor_drift(
    reference_path: Path,
    current_path: Path,
    report_path: Path,
    threshold: float = 0.2,
    target: str | None = None,
) -> dict[str, Any]:
    """Compare feature distributions using PSI for numeric and categorical data."""
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")

    reference = _read_csv(reference_path)
    current = _read_csv(current_path)
    if target is not None:
        if target not in reference.columns:
            raise ValueError(
                f"Target column '{target}' was not found in {reference_path}"
            )
        reference = reference.drop(columns=[target])
        if target in current.columns:
            current = current.drop(columns=[target])
    feature_columns = reference.columns.tolist()
    _validate_features(reference, feature_columns)
    _validate_features(current, feature_columns)

    feature_results: dict[str, Any] = {}
    epsilon = 1e-6
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            baseline = reference[column].dropna().to_numpy(dtype=float)
            observed = pd.to_numeric(current[column], errors="coerce")
            invalid_values = current[column].notna() & observed.isna()
            if invalid_values.any():
                raise ValueError(
                    f"Current data contains non-numeric values for feature '{column}'"
                )
            observed_values = observed.dropna().to_numpy(dtype=float)
            quantiles = np.quantile(baseline, np.linspace(0, 1, 11))
            edges = np.unique(np.concatenate(([-np.inf], quantiles, [np.inf])))
            reference_counts = np.histogram(baseline, bins=edges)[0].astype(float)
            current_counts = np.histogram(observed_values, bins=edges)[0].astype(float)
            method = "population_stability_index_numeric"
        else:
            reference_values = reference[column].astype("string").fillna("<MISSING>")
            current_values = current[column].astype("string").fillna("<MISSING>")
            categories = sorted(
                set(reference_values.tolist()) | set(current_values.tolist())
            )
            reference_counts = (
                reference_values.value_counts(normalize=False)
                .reindex(categories, fill_value=0)
                .to_numpy(dtype=float)
            )
            current_counts = (
                current_values.value_counts(normalize=False)
                .reindex(categories, fill_value=0)
                .to_numpy(dtype=float)
            )
            method = "population_stability_index_categorical"

        reference_distribution = (reference_counts + epsilon) / (
            reference_counts.sum() + epsilon * len(reference_counts)
        )
        current_distribution = (current_counts + epsilon) / (
            current_counts.sum() + epsilon * len(current_counts)
        )
        score = float(
            np.sum(
                (current_distribution - reference_distribution)
                * np.log(current_distribution / reference_distribution)
            )
        )
        feature_results[column] = {
            "method": method,
            "score": score,
            "drift_detected": score >= threshold,
        }

    report = {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "threshold": threshold,
        "features": feature_results,
        "drift_detected": any(
            result["drift_detected"] for result in feature_results.values()
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_models(
    csv_path: Path,
    target: str,
    model_path: Path,
    metrics_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    experiments_path: Path | None = Path("artifacts/experiments.jsonl"),
    experiment_name: str = "classification-comparison",
    version: str | None = None,
    mlflow_tracking: bool = False,
) -> dict[str, Any]:
    """Train a small set of candidate models and save the best performer."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    data = _read_csv(csv_path)
    if data.empty:
        raise ValueError(f"Training data is empty: {csv_path}")
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found in {csv_path}")
    if data[target].isna().any():
        raise ValueError(f"Target column '{target}' contains missing values")

    features = data.drop(columns=[target])
    labels = data[target]
    if features.shape[1] == 0:
        raise ValueError("Training data must contain at least one feature column")
    _validate_features(features, features.columns.tolist())
    if labels.nunique() < 2:
        raise ValueError("Classification requires at least two target classes")
    class_counts = labels.value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class must have at least two rows")

    test_rows = math.ceil(len(data) * test_size)
    train_rows = len(data) - test_rows
    if test_rows < labels.nunique() or train_rows < labels.nunique():
        raise ValueError(
            "The train/test split must have at least one row per class; "
            "add more rows or adjust test_size"
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    candidate_models = {
        "logistic_regression": _build_model(x_train),
        "random_forest": Pipeline(
            [
                ("preprocessing", _build_model(x_train).named_steps["preprocessing"]),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    effective_cv = min(5, int(y_train.value_counts().min()))
    if effective_cv < 2:
        raise ValueError(
            "Model comparison needs at least two training rows for every class"
        )

    evaluations: list[dict[str, Any]] = []
    for model_name, model in candidate_models.items():
        cv_scores = cross_val_score(
            model,
            x_train,
            y_train,
            scoring="f1_macro",
            cv=effective_cv,
            n_jobs=1,
        )
        evaluations.append(
            {
                **_evaluate_model(model, x_train, x_test, y_train, y_test, model_name),
                "cv_f1_macro": float(cv_scores.mean()),
            }
        )

    best_model = max(evaluations, key=lambda result: result["cv_f1_macro"])
    selected_model = candidate_models[best_model["model_name"]]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_version = version or _create_model_version("comparison")
    artifact_bundle = {
        "model": selected_model,
        "features": features.columns.tolist(),
        "target": target,
        "model_name": best_model["model_name"],
        "experiment_name": experiment_name,
        "model_version": resolved_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    _persist_model_artifact(model_path, artifact_bundle, resolved_version)

    metrics_payload = {
        "best_model": best_model["model_name"],
        "best_accuracy": best_model["accuracy"],
        "best_precision_macro": best_model["precision_macro"],
        "best_recall_macro": best_model["recall_macro"],
        "best_f1": best_model["f1_macro"],
        "best_cv_f1_macro": best_model["cv_f1_macro"],
        "best_classification_report": best_model["classification_report"],
        "best_confusion_matrix": best_model["confusion_matrix"],
        "selection_metric": "cv_f1_macro",
        "selection_cv_folds": effective_cv,
        "comparison": evaluations,
        "target": target,
        "features": features.columns.tolist(),
        "random_state": random_state,
        "test_size": test_size,
        "model_version": resolved_version,
        "dataset_sha256": artifact_bundle["dataset_sha256"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    if experiments_path is not None:
        run = {
            "run_id": str(uuid.uuid4()),
            "experiment": experiment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": str(csv_path),
            "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "parameters": {
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
            },
            "metrics": {
                "best_model": best_model["model_name"],
                "best_accuracy": best_model["accuracy"],
                "best_f1": best_model["f1"],
            },
            "model_path": str(model_path),
            "model_version": resolved_version,
            "metrics_path": str(metrics_path),
        }
        experiments_path.parent.mkdir(parents=True, exist_ok=True)
        with experiments_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(run) + "\n")
    if mlflow_tracking:
        mlflow_metrics = {}
        for evaluation in evaluations:
            model_name = evaluation["model_name"]
            for metric_name in (
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "cv_f1_macro",
            ):
                mlflow_metrics[f"{model_name}_{metric_name}"] = evaluation[metric_name]
        _log_mlflow_run(
            experiment_name,
            {
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "dataset_sha256": artifact_bundle["dataset_sha256"],
                "selected_model": best_model["model_name"],
                "model_version": resolved_version,
            },
            mlflow_metrics,
            [model_path, metrics_path],
        )
    return metrics_payload


def train(
    csv_path: Path,
    target: str,
    model_path: Path,
    metrics_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    tune: bool = True,
    cv_folds: int = 5,
    experiments_path: Path | None = Path("artifacts/experiments.jsonl"),
    experiment_name: str = "classification",
    version: str | None = None,
    mlflow_tracking: bool = False,
) -> dict[str, Any]:
    """Validate, tune, evaluate, and persist a classification pipeline."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    data = _read_csv(csv_path)
    if data.empty:
        raise ValueError(f"Training data is empty: {csv_path}")
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found in {csv_path}")
    if data[target].isna().any():
        raise ValueError(f"Target column '{target}' contains missing values")

    features = data.drop(columns=[target])
    labels = data[target]
    if features.shape[1] == 0:
        raise ValueError("Training data must contain at least one feature column")
    _validate_features(features, features.columns.tolist())
    if labels.nunique() < 2:
        raise ValueError("Classification requires at least two target classes")
    class_counts = labels.value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class must have at least two rows")
    test_rows = math.ceil(len(data) * test_size)
    train_rows = len(data) - test_rows
    if test_rows < labels.nunique() or train_rows < labels.nunique():
        raise ValueError(
            "The train/test split must have at least one row per class; "
            "add more rows or adjust test_size"
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    model = _build_model(x_train)
    effective_cv = min(cv_folds, int(y_train.value_counts().min()))
    if tune and effective_cv < 2:
        raise ValueError(
            "Cross-validation needs at least two training rows for every class"
        )
    best_params: dict[str, Any] = {}
    cv_accuracy: float | None = None
    if tune:
        search = GridSearchCV(
            model,
            param_grid={"classifier__C": [0.1, 1.0, 10.0]},
            scoring="accuracy",
            cv=effective_cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_accuracy = float(search.best_score_)
    else:
        model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    evaluation_labels = pd.unique(y_test).tolist()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": {
            "labels": evaluation_labels,
            "values": confusion_matrix(
                y_test, predictions, labels=evaluation_labels
            ).tolist(),
        },
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "target": target,
        "features": features.columns.tolist(),
        "test_rows": len(y_test),
        "random_state": random_state,
        "tuning_enabled": tune,
        "cv_folds": effective_cv if tune else None,
        "cv_accuracy": cv_accuracy,
        "best_params": best_params,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_version = version or _create_model_version("model")
    artifact_bundle = {
        "model": model,
        "features": features.columns.tolist(),
        "target": target,
        "experiment_name": experiment_name,
        "model_version": resolved_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    _persist_model_artifact(model_path, artifact_bundle, resolved_version)
    metrics["model_version"] = resolved_version
    metrics["dataset_sha256"] = artifact_bundle["dataset_sha256"]
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if experiments_path is not None:
        run = {
            "run_id": str(uuid.uuid4()),
            "experiment": experiment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": str(csv_path),
            "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "parameters": {
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "tuning_enabled": tune,
                "cv_folds": effective_cv if tune else None,
                **best_params,
            },
            "metrics": {
                "accuracy": metrics["accuracy"],
                "cv_accuracy": cv_accuracy,
            },
            "model_path": str(model_path),
            "model_version": resolved_version,
            "metrics_path": str(metrics_path),
        }
        experiments_path.parent.mkdir(parents=True, exist_ok=True)
        with experiments_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(run) + "\n")
    if mlflow_tracking:
        _log_mlflow_run(
            experiment_name,
            {
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "tuning_enabled": tune,
                "cv_folds": effective_cv if tune else None,
                "dataset_sha256": artifact_bundle["dataset_sha256"],
                "model_version": resolved_version,
                **best_params,
            },
            {
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "cv_accuracy": cv_accuracy,
            },
            [model_path, metrics_path],
        )
    return metrics


def predict(model_path: Path, csv_path: Path, output_path: Path) -> pd.DataFrame:
    """Apply a saved pipeline to a CSV and write predictions beside each row."""
    bundle = joblib.load(model_path)
    data = _read_csv(csv_path)
    data = predict_frame(bundle, data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run an end-to-end tabular classification pipeline."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and evaluate a model")
    train_parser.add_argument("--data", type=Path, required=True, help="training CSV")
    train_parser.add_argument("--target", required=True, help="target column name")
    train_parser.add_argument(
        "--model", type=Path, default=Path("artifacts/model.joblib")
    )
    train_parser.add_argument(
        "--metrics", type=Path, default=Path("artifacts/metrics.json")
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument(
        "--no-tune",
        action="store_true",
        help="skip cross-validated hyperparameter tuning",
    )
    train_parser.add_argument("--cv-folds", type=int, default=5)
    train_parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("artifacts/experiments.jsonl"),
        help="JSONL experiment log",
    )
    train_parser.add_argument(
        "--no-experiment-log",
        action="store_true",
        help="do not append a run to the experiment log",
    )
    train_parser.add_argument(
        "--experiment-name", default="classification", help="experiment group name"
    )
    train_parser.add_argument(
        "--version",
        default=None,
        help="optional model version label; also saves a versioned artifact copy",
    )
    train_parser.add_argument(
        "--mlflow",
        action="store_true",
        help="log parameters, metrics, and model artifacts to MLflow",
    )

    compare_parser = commands.add_parser(
        "compare",
        help="train a few candidate models and keep the best one",
    )
    compare_parser.add_argument("--data", type=Path, required=True, help="training CSV")
    compare_parser.add_argument("--target", required=True, help="target column name")
    compare_parser.add_argument(
        "--model", type=Path, default=Path("artifacts/best_model.joblib")
    )
    compare_parser.add_argument(
        "--metrics", type=Path, default=Path("artifacts/model_comparison.json")
    )
    compare_parser.add_argument("--test-size", type=float, default=0.2)
    compare_parser.add_argument("--random-state", type=int, default=42)
    compare_parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("artifacts/experiments.jsonl"),
        help="JSONL experiment log",
    )
    compare_parser.add_argument(
        "--no-experiment-log",
        action="store_true",
        help="do not append a run to the experiment log",
    )
    compare_parser.add_argument(
        "--experiment-name",
        default="classification-comparison",
        help="experiment group name",
    )
    compare_parser.add_argument(
        "--version",
        default=None,
        help="optional model version label; also saves a versioned artifact copy",
    )
    compare_parser.add_argument(
        "--mlflow",
        action="store_true",
        help="log candidate metrics and artifacts to MLflow",
    )

    monitor_parser = commands.add_parser(
        "monitor",
        help="compare reference and current feature distributions for drift",
    )
    monitor_parser.add_argument(
        "--reference", type=Path, required=True, help="baseline feature CSV"
    )
    monitor_parser.add_argument(
        "--current", type=Path, required=True, help="new feature CSV to monitor"
    )
    monitor_parser.add_argument(
        "--report", type=Path, default=Path("artifacts/drift_report.json")
    )
    monitor_parser.add_argument("--threshold", type=float, default=0.2)
    monitor_parser.add_argument(
        "--target",
        default=None,
        help="optional target column in the reference data to exclude from monitoring",
    )

    predict_parser = commands.add_parser(
        "predict", help="generate predictions with a saved model"
    )
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--data", type=Path, required=True, help="input CSV")
    predict_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        metrics = train(
            args.data,
            args.target,
            args.model,
            args.metrics,
            args.test_size,
            args.random_state,
            not args.no_tune,
            args.cv_folds,
            None if args.no_experiment_log else args.experiments,
            args.experiment_name,
            args.version,
            args.mlflow,
        )
        print(f"Model saved to {args.model}")
        if args.version:
            print(f"Versioned artifact saved with version {args.version}")
        print(f"Metrics saved to {args.metrics}")
        print(f"Holdout accuracy: {metrics['accuracy']:.4f}")
    elif args.command == "compare":
        metrics = compare_models(
            args.data,
            args.target,
            args.model,
            args.metrics,
            args.test_size,
            args.random_state,
            None if args.no_experiment_log else args.experiments,
            args.experiment_name,
            args.version,
            args.mlflow,
        )
        print(f"Best model saved to {args.model}: {metrics['best_model']}")
        if args.version:
            print(f"Versioned artifact saved with version {args.version}")
        print(f"Metrics saved to {args.metrics}")
        print(f"Best F1 score: {metrics['best_f1']:.4f}")
    elif args.command == "predict":
        result = predict(args.model, args.data, args.output)
        print(f"Wrote {len(result)} predictions to {args.output}")
    elif args.command == "monitor":
        report = monitor_drift(
            args.reference,
            args.current,
            args.report,
            args.threshold,
            args.target,
        )
        print(f"Drift report saved to {args.report}")
        print(
            f"Drift detected: {report['drift_detected']} "
            f"({sum(item['drift_detected'] for item in report['features'].values())} "
            f"features over threshold)"
        )


if __name__ == "__main__":
    main()
