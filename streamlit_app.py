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
st.write(
    "Upload a CSV to get a prediction for each row. This demo predicts a category "
    "such as `yes` or `no`; it does not change your input file."
)
st.warning(
    "The included student-outcome dataset is only a small demonstration. Its "
    "scores are not proof that predictions will be accurate for real students."
)

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
        except (OSError, ValueError, EOFError, KeyError) as exc:
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
    except (OSError, ValueError, EOFError, KeyError) as exc:
        st.session_state["bundle"] = None
        st.error(f"Could not load the default model: {exc}")

bundle = st.session_state.get("bundle")
if bundle is None:
    st.info("Load a model to begin.")
    st.stop()

model_path = Path(st.session_state.get("selected_model", selected_model))
metrics_summary = _load_metrics_for_model(model_path)

features = bundle.get("features", [])
target = bundle.get("target", "target")
st.subheader("What this model predicts")
st.write(f"**Prediction:** `{target}`")
st.write("**Required input columns:** " + ", ".join(f"`{name}`" for name in features))
st.caption(
    "Include one record per row and all required columns. Extra columns are ignored. "
    "The output adds a `prediction` column."
)

with st.expander("Technical model details"):
    st.json(
        {
            "model_name": bundle.get("model_name", type(bundle["model"]).__name__),
            "model_version": bundle.get("model_version"),
            "created_at": bundle.get("created_at"),
            "dataset_sha256": bundle.get("dataset_sha256"),
        }
    )

if metrics_summary:
    st.subheader("How well did the model do on its test data?")
    st.caption(
        "These scores summarize earlier test examples; they do not guarantee the "
        "same accuracy on new data. A score of 100% means all examples in that "
        "test set were predicted correctly."
    )
    if "comparison" in metrics_summary:
        comparison_rows = []
        for row in metrics_summary["comparison"]:
            comparison_rows.append(
                {
                    "model": row.get("model_name"),
                    "Test accuracy": f"{row.get('accuracy', 0):.0%}",
                    "Macro precision": (
                        f"{row.get('precision_macro', row.get('precision', 0)):.0%}"
                    ),
                    "Macro recall": (
                        f"{row.get('recall_macro', row.get('recall', 0)):.0%}"
                    ),
                    "Macro F1": f"{row.get('f1_macro', row.get('f1', 0)):.0%}",
                    "Cross-validation F1": (
                        f"{row['cv_f1_macro']:.0%} ± "
                        f"{row.get('cv_f1_macro_std', 0):.0%}"
                        if row.get("cv_f1_macro") is not None
                        else "Not available"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
        st.caption(
            "Accuracy is the share of test rows predicted correctly. Macro precision, "
            "recall, and F1 average the score across classes. Cross-validation F1 "
            "shows the mean ± standard deviation across repeated training folds; "
            "it was used to select the model."
        )
        with st.expander("Best model class-level metrics"):
            st.dataframe(
                pd.DataFrame.from_dict(
                    metrics_summary.get("best_classification_report", {}),
                    orient="index",
                ),
                use_container_width=True,
            )
            confusion = metrics_summary.get("best_confusion_matrix", {})
            labels = confusion.get("labels", [])
            values = confusion.get("values", [])
            if labels and values:
                st.caption(
                    "Confusion matrix: rows are actual classes; columns are predicted "
                    "classes."
                )
                st.dataframe(
                    pd.DataFrame(
                        values,
                        index=[f"Actual: {label}" for label in labels],
                        columns=[f"Predicted: {label}" for label in labels],
                    ),
                    use_container_width=True,
                )
    elif "accuracy" in metrics_summary:
        test_rows = metrics_summary.get("test_rows")
        help_text = (
            f"Measured on {test_rows} held-out rows."
            if test_rows is not None
            else "Measured on held-out test rows."
        )
        metric_columns = st.columns(4)
        for column, (label, key) in zip(
            metric_columns,
            [
                ("Test accuracy", "accuracy"),
                ("Macro precision", "precision_macro"),
                ("Macro recall", "recall_macro"),
                ("Macro F1", "f1_macro"),
            ],
        ):
            value = metrics_summary.get(key)
            column.metric(
                label,
                f"{value:.0%}" if value is not None else "Not available",
                help=help_text,
            )
        st.caption(
            "Accuracy is the share of held-out test rows the model predicted "
            "correctly. Precision, recall, and F1 provide additional views of "
            "performance across the predicted classes."
        )
        if metrics_summary.get("cv_accuracy") is not None:
            cv_std = metrics_summary.get("cv_accuracy_std")
            variability = f" ± {cv_std:.0%}" if cv_std is not None else ""
            repeats = metrics_summary.get("cv_repeats")
            repeat_text = f" across {repeats} repeats" if repeats is not None else ""
            st.caption(
                f"Training cross-validation accuracy: "
                f"{metrics_summary['cv_accuracy']:.0%}{variability}"
                f"{repeat_text}. The ± value is the standard deviation across folds."
            )
    else:
        st.info("No familiar evaluation metrics were found for this model.")
else:
    st.info(
        "No metrics report was found next to this model. Predictions still work, "
        "but model quality scores are not available here."
    )

st.subheader("Make predictions")
template = pd.DataFrame(columns=features)
st.download_button(
    "Download an empty CSV template",
    data=template.to_csv(index=False),
    file_name="prediction_template.csv",
    mime="text/csv",
    help="The template contains the required column names. Add one case per row.",
)
uploaded_file = st.file_uploader(
    "Choose a CSV file with one case per row", type=["csv"]
)
if hasattr(bundle["model"], "predict_proba"):
    st.caption(
        "The output also includes a confidence score from the model. Treat it as "
        "a relative model score, not a guarantee or a calibrated probability."
    )

if uploaded_file is not None:
    try:
        frame = pd.read_csv(uploaded_file)
        result = predict_frame(bundle, frame)
        result = _attach_confidence(bundle, result)
        st.subheader("Predictions")
        st.caption(
            "`prediction` is the model's predicted category for that row. "
            "`confidence`, when present, is the model's score for its predicted "
            "category; it is not a guarantee."
        )
        st.dataframe(result, use_container_width=True)

        csv_buffer = BytesIO()
        result.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )
    except (ValueError, pd.errors.ParserError, OSError) as exc:
        st.error(f"Prediction failed: {exc}")
else:
    st.info("Choose a CSV file to generate predictions.")
