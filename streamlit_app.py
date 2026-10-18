from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ml_pipeline import predict_frame

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


@st.cache_resource
def load_model(model_path: Path) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or not {"model", "features", "target"}.issubset(
        bundle
    ):
        raise ValueError(f"Model artifact has an invalid format: {model_path}")
    return bundle


def _find_model_choices() -> list[str]:
    artifact_dir = Path("artifacts")
    if artifact_dir.exists():
        choices = sorted(str(path) for path in artifact_dir.glob("*.joblib"))
        if choices:
            return choices
    return [str(DEFAULT_MODEL_PATH)]


def _load_metrics_for_model(model_path: Path) -> dict[str, Any]:
    candidates = [
        model_path.parent / "metrics.json",
        model_path.parent / "model_comparison.json",
        model_path.parent / "artifacts" / "metrics.json",
        model_path.parent / "artifacts" / "model_comparison.json",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {}


def _attach_confidence(bundle: dict[str, Any], result: pd.DataFrame) -> pd.DataFrame:
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        return result
    features = bundle["features"]
    probabilities = model.predict_proba(result[features])
    class_labels = list(model.classes_)
    confidence_values = [
        float(probabilities[index][class_labels.index(prediction)])
        for index, prediction in enumerate(result["prediction"])
    ]
    result = result.copy()
    result["confidence"] = confidence_values
    return result


st.set_page_config(page_title="ML Pipeline Dashboard", page_icon="📊", layout="wide")
st.title("End-to-End ML Pipeline Dashboard")

with st.sidebar:
    st.header("Model")
    model_choices = _find_model_choices()
    selected_model = st.selectbox("Choose a model artifact", options=model_choices)
    if st.button("Load selected model"):
        try:
            model_bundle = load_model(Path(selected_model))
            st.session_state["bundle"] = model_bundle
            st.session_state["selected_model"] = selected_model
            model_name = model_bundle.get(
                "model_name", type(model_bundle["model"]).__name__
            )
            st.success(f"Loaded model: {model_name}")
        except Exception as exc:  # pragma: no cover - UI feedback only
            st.error(f"Could not load model: {exc}")

    if "selected_model" in st.session_state:
        st.caption(f"Current file: {st.session_state['selected_model']}")

if "bundle" not in st.session_state:
    try:
        st.session_state["bundle"] = load_model(
            Path(
                selected_model
                if "selected_model" not in st.session_state
                else st.session_state["selected_model"]
            )
        )
    except Exception:
        st.session_state["bundle"] = None

bundle = st.session_state.get("bundle")
if bundle is None:
    st.info("Load a model to begin.")
    st.stop()

model_path = Path(st.session_state.get("selected_model", selected_model))
metrics_summary = _load_metrics_for_model(model_path)

st.subheader("Model metadata")
metadata = {
    "target": bundle.get("target"),
    "features": bundle.get("features", []),
    "model_name": bundle.get("model_name", type(bundle["model"]).__name__),
    "model_version": bundle.get("model_version"),
    "created_at": bundle.get("created_at"),
    "dataset_sha256": bundle.get("dataset_sha256"),
}
st.json(metadata)

if metrics_summary:
    st.subheader("Model metrics summary")
    if "comparison" in metrics_summary:
        comparison_rows = []
        for row in metrics_summary["comparison"]:
            comparison_rows.append(
                {
                    "model": row.get("model_name"),
                    "accuracy": row.get("accuracy"),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "f1": row.get("f1"),
                }
            )
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
    else:
        summary = {
            key: value
            for key, value in metrics_summary.items()
            if key
            in {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "cv_accuracy",
                "best_accuracy",
                "best_f1",
            }
        }
        st.json(summary)

left_column, right_column = st.columns([1.5, 1])
with left_column:
    uploaded_file = st.file_uploader("Upload a CSV file for predictions", type=["csv"])

with right_column:
    st.caption(
        "Prediction output includes a confidence score when the model supports "
        "probabilities."
    )

if uploaded_file is not None:
    try:
        frame = pd.read_csv(uploaded_file)
        result = predict_frame(bundle, frame)
        result = _attach_confidence(bundle, result)
        st.subheader("Predictions")
        st.dataframe(result, use_container_width=True)

        csv_buffer = BytesIO()
        result.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:  # pragma: no cover - UI feedback only
        st.error(f"Prediction failed: {exc}")
else:
    st.info("Choose a CSV file to generate predictions.")
