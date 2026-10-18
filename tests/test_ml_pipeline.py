import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient

from api import create_app
from ml_pipeline import (
    _log_mlflow_run,
    compare_models,
    monitor_drift,
    predict,
    train,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.data_path = self.directory / "train.csv"
        rows = []
        for index in range(60):
            label = "high" if index % 2 else "low"
            rows.append(
                {
                    "measurement": index + (20 if label == "high" else 0),
                    "group": "east" if index % 3 else "west",
                    "label": label,
                }
            )
        pd.DataFrame(rows).to_csv(self.data_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_save_and_predict(self):
        model_path = self.directory / "artifacts" / "model.joblib"
        metrics_path = self.directory / "artifacts" / "metrics.json"
        metrics = train(
            self.data_path,
            "label",
            model_path,
            metrics_path,
            random_state=7,
            experiments_path=self.directory / "experiments.jsonl",
        )

        self.assertTrue(model_path.exists())
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["features"], ["measurement", "group"])
        self.assertGreaterEqual(metrics["cv_folds"], 2)
        self.assertIn("classifier__C", metrics["best_params"])
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(set(metrics["confusion_matrix"]["labels"]), {"low", "high"})
        self.assertIn("macro avg", metrics["classification_report"])
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(saved_metrics["accuracy"], metrics["accuracy"])
        experiment = json.loads(
            (self.directory / "experiments.jsonl").read_text(encoding="utf-8")
        )
        self.assertEqual(experiment["experiment"], "classification")
        self.assertEqual(experiment["metrics"]["accuracy"], metrics["accuracy"])

        prediction_input = self.directory / "predict.csv"
        pd.DataFrame(
            [
                {"measurement": 5, "group": "north"},
                {"measurement": 85, "group": "east"},
            ]
        ).to_csv(prediction_input, index=False)
        output_path = self.directory / "predictions.csv"
        result = predict(model_path, prediction_input, output_path)

        self.assertEqual(result["prediction"].tolist(), ["low", "high"])
        self.assertEqual(result["group"].tolist(), ["north", "east"])
        self.assertTrue(output_path.exists())

    def test_compare_models_picks_best_candidate(self):
        model_path = self.directory / "comparison-model.joblib"
        metrics_path = self.directory / "comparison-metrics.json"
        metrics = compare_models(
            self.data_path,
            "label",
            model_path,
            metrics_path,
            experiments_path=None,
            version="v-test",
        )
        self.assertTrue(model_path.exists())
        self.assertTrue(
            model_path.with_name(
                f"{model_path.stem}-v-test{model_path.suffix}"
            ).exists()
        )
        self.assertEqual(len(metrics["comparison"]), 2)
        self.assertIn(metrics["best_model"], {"logistic_regression", "random_forest"})
        self.assertGreaterEqual(metrics["best_f1"], 0.0)
        self.assertEqual(metrics["selection_metric"], "cv_f1_macro")
        self.assertGreaterEqual(metrics["selection_cv_folds"], 2)
        self.assertIn("best_confusion_matrix", metrics)
        self.assertIn("best_classification_report", metrics)
        for candidate in metrics["comparison"]:
            self.assertIn("cv_f1_macro", candidate)
            self.assertEqual(
                set(candidate["confusion_matrix"]["labels"]), {"low", "high"}
            )
            self.assertIn("macro avg", candidate["classification_report"])

    def test_monitor_drift_writes_feature_report(self):
        current_path = self.directory / "current.csv"
        pd.DataFrame(
            [
                {"measurement": 150, "group": "north"},
                {"measurement": 160, "group": "north"},
                {"measurement": 170, "group": "south"},
            ]
        ).to_csv(current_path, index=False)
        report_path = self.directory / "drift.json"

        report = monitor_drift(
            self.data_path, current_path, report_path, target="label"
        )

        self.assertTrue(report_path.exists())
        self.assertEqual(report["reference_rows"], 60)
        self.assertEqual(report["current_rows"], 3)
        self.assertTrue(report["drift_detected"])
        self.assertEqual(set(report["features"]), {"measurement", "group"})
        self.assertTrue(report["features"]["measurement"]["drift_detected"])
        self.assertEqual(
            report["features"]["group"]["method"],
            "population_stability_index_categorical",
        )

    def test_monitor_drift_rejects_missing_feature(self):
        current_path = self.directory / "missing-feature.csv"
        pd.DataFrame([{"measurement": 5}]).to_csv(current_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing required feature columns"):
            monitor_drift(
                self.data_path,
                current_path,
                self.directory / "drift.json",
                target="label",
            )

    def test_monitor_drift_rejects_non_numeric_current_value(self):
        current_path = self.directory / "invalid-numeric.csv"
        pd.DataFrame([{"measurement": "not-a-number", "group": "east"}]).to_csv(
            current_path, index=False
        )
        with self.assertRaisesRegex(ValueError, "non-numeric values"):
            monitor_drift(
                self.data_path,
                current_path,
                self.directory / "drift.json",
                target="label",
            )

    def test_mlflow_is_optional_and_reports_install_instructions(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-mlflow.txt"):
                _log_mlflow_run("test", {}, {}, [])

    def test_mlflow_tracking_logs_metrics_and_artifacts(self):
        mlflow_stub = MagicMock()
        artifact_path = self.directory / "report.json"
        artifact_path.write_text("{}", encoding="utf-8")
        with patch.dict(sys.modules, {"mlflow": mlflow_stub}):
            _log_mlflow_run(
                "test-experiment",
                {"folds": 3, "options": {"tune": True}},
                {"macro_f1": 0.9},
                [artifact_path],
            )

        mlflow_stub.set_experiment.assert_called_once_with("test-experiment")
        mlflow_stub.log_params.assert_called_once_with(
            {"folds": "3", "options": '{"tune": true}'}
        )
        mlflow_stub.log_metrics.assert_called_once_with({"macro_f1": 0.9})
        mlflow_stub.log_artifact.assert_called_once_with(str(artifact_path))

    def test_missing_target_column_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Target column 'absent'"):
            train(
                self.data_path,
                "absent",
                self.directory / "model.joblib",
                self.directory / "metrics.json",
            )

    def test_prediction_requires_all_training_features(self):
        model_path = self.directory / "model.joblib"
        train(
            self.data_path,
            "label",
            model_path,
            self.directory / "metrics.json",
            experiments_path=None,
        )
        incomplete_input = self.directory / "incomplete.csv"
        pd.DataFrame([{"measurement": 10}]).to_csv(incomplete_input, index=False)
        with self.assertRaisesRegex(ValueError, "missing required feature columns"):
            predict(model_path, incomplete_input, self.directory / "out.csv")

    def test_training_rejects_feature_column_with_no_values(self):
        data = pd.read_csv(self.data_path)
        data["empty"] = None
        invalid_path = self.directory / "empty-feature.csv"
        data.to_csv(invalid_path, index=False)
        with self.assertRaisesRegex(ValueError, "only missing values"):
            train(
                invalid_path,
                "label",
                self.directory / "model.joblib",
                self.directory / "metrics.json",
            )

    def test_training_rejects_duplicate_csv_headers(self):
        invalid_path = self.directory / "duplicate-headers.csv"
        invalid_path.write_text(
            "measurement,measurement,label\n1,2,low\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            train(
                invalid_path,
                "label",
                self.directory / "model.joblib",
                self.directory / "metrics.json",
            )

    def test_training_rejects_csv_without_header(self):
        invalid_path = self.directory / "empty.csv"
        invalid_path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must include a header row"):
            train(
                invalid_path,
                "label",
                self.directory / "model.joblib",
                self.directory / "metrics.json",
            )

    def test_prediction_api_serves_records_and_validates_schema(self):
        model_path = self.directory / "api-model.joblib"
        train(
            self.data_path,
            "label",
            model_path,
            self.directory / "api-metrics.json",
            experiments_path=None,
        )
        with TestClient(create_app(model_path)) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            metadata_response = client.get("/metadata")
            self.assertEqual(metadata_response.status_code, 200)
            self.assertEqual(metadata_response.json()["target"], "label")
            self.assertEqual(
                metadata_response.json()["features"], ["measurement", "group"]
            )
            response = client.post(
                "/predict",
                json={
                    "records": [
                        {"measurement": 5, "group": "north"},
                        {"measurement": 85, "group": "east"},
                    ]
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["predictions"], ["low", "high"])
            details = response.json()["details"]
            self.assertEqual(len(details), 2)
            self.assertEqual(details[0]["prediction"], "low")
            self.assertIn("confidence", details[0])
            self.assertTrue(0.0 <= float(details[0]["confidence"]) <= 1.0)
            invalid_response = client.post(
                "/predict", json={"records": [{"measurement": 5}]}
            )
            self.assertEqual(invalid_response.status_code, 422)
            empty_records_response = client.post("/predict", json={"records": []})
            self.assertEqual(empty_records_response.status_code, 422)
            malformed_payload_response = client.post(
                "/predict", json={"records": "not-a-list"}
            )
            self.assertEqual(malformed_payload_response.status_code, 422)
            invalid_value_response = client.post(
                "/predict",
                json={"records": [{"measurement": "not-numeric", "group": "north"}]},
            )
            self.assertEqual(invalid_value_response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
