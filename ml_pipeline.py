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
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
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
    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


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
    evaluations: list[dict[str, Any]] = []
    for model_name, model in candidate_models.items():
        evaluations.append(
            _evaluate_model(model, x_train, x_test, y_train, y_test, model_name)
        )

    best_model = max(evaluations, key=lambda result: result["f1"])
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
        "best_f1": best_model["f1"],
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

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
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
        )
        print(f"Best model saved to {args.model}: {metrics['best_model']}")
        if args.version:
            print(f"Versioned artifact saved with version {args.version}")
        print(f"Metrics saved to {args.metrics}")
        print(f"Best F1 score: {metrics['best_f1']:.4f}")
    elif args.command == "predict":
        result = predict(args.model, args.data, args.output)
        print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
