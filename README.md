# End-to-end tabular classification pipeline

[![CI](https://github.com/subhamkumarswain/End-to-End_ML_Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/subhamkumarswain/End-to-End_ML_Pipeline/actions/workflows/ci.yml)

An end-to-end, reproducible **tabular classification** project: validate CSV
data, compare and evaluate scikit-learn models, version trained artifacts, and
serve predictions through a REST API or interactive Streamlit dashboard.

It is intentionally a tabular **classification** pipeline, not a universal
solution for image, text, time-series, or regression problems. The included
student-outcome dataset is a small demonstration dataset; its results should not
be treated as a production model or evidence of real-world predictive quality.

**Workflow:** CSV validation → stratified split → preprocessing and model
selection → holdout evaluation → versioned model + metrics → prediction,
tracking, and drift checks.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python ml_pipeline.py compare --data .\data\train.csv --target passed --model .\artifacts\demo_model.joblib --metrics .\artifacts\demo_metrics.json --version demo-v1 --no-experiment-log
python -m streamlit run streamlit_app.py
```

Then open the Streamlit URL printed in the terminal (usually
`http://localhost:8501`).

## Setup

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

The included `data\train.csv` and `data\predict.csv` make the example runnable.
The training CSV must contain the target column and at least one feature.
Numeric features are median-imputed and scaled; categorical features are
imputed and one-hot encoded. Completely empty feature columns and missing
targets are rejected. The default split is 80% training and 20% stratified
holdout data. By default, the pipeline searches logistic-regression `C` values
using stratified cross-validation on the training split; the holdout split is
not used for tuning.

```powershell
python ml_pipeline.py train --data .\data\train.csv --target passed
```

Training writes `artifacts\model.joblib`, `artifacts\metrics.json`, and appends
the run parameters, dataset SHA-256, CV score, and holdout score to
`artifacts\experiments.jsonl`. These paths, cross-validation folds, holdout
fraction, and random seed can be changed:

```powershell
python ml_pipeline.py train `
  --data .\data\train.csv `
  --target passed `
  --model .\artifacts\classifier.joblib `
  --metrics .\artifacts\evaluation.json `
  --experiments .\artifacts\experiments.jsonl `
  --cv-folds 4 `
  --test-size 0.25 `
  --random-state 17
```

Use `--no-tune` to skip tuning, or `--no-experiment-log` to disable the local
JSON Lines tracker.

To compare a few candidate models and keep the best performer, run:

```powershell
python ml_pipeline.py compare `
  --data .\data\train.csv `
  --target passed `
  --model .\artifacts\best_model.joblib `
  --metrics .\artifacts\model_comparison.json `
  --version v1
```

The comparison selects the model using training-only stratified cross-validation
macro F1, then reports holdout accuracy, macro precision/recall/F1, per-class
metrics, and confusion matrices. This avoids choosing a model based on its
holdout results. It keeps the highest cross-validation macro-F1 model.
When a version is supplied, it also saves a versioned artifact copy such as
`best_model-v1.joblib`, while preserving the latest model at the regular path.

## Experiment tracking with MLflow (optional)

Install the optional tracking dependency and enable logging on training or model
comparison:

```powershell
python -m pip install -r requirements-mlflow.txt
python ml_pipeline.py compare --data .\data\train.csv --target passed --mlflow
mlflow ui
```

Runs are stored locally in `mlruns/` by default and include parameters, scores,
the saved model artifact, and the metrics report. Configure a remote MLflow
tracking server with the `MLFLOW_TRACKING_URI` environment variable. Do not put
credentials in source control.

## Feature drift monitoring

Compare a new unlabeled CSV with the training feature distribution:

```powershell
python ml_pipeline.py monitor `
  --reference .\data\train.csv `
  --current .\data\predict.csv `
  --target passed `
  --report .\artifacts\drift_report.json
```

The report uses population stability index (PSI) per feature, with separate
numeric and categorical handling. The default alert threshold is 0.2 and can be
changed with `--threshold`. Drift is a signal to investigate, not proof that
model accuracy has degraded; labeled recent outcomes are needed to measure
performance.

## Predict

The prediction CSV must include every feature column used during training. Extra
columns are ignored, and previously unseen categorical values are supported.
The output CSV contains the input columns plus a `prediction` column.

```powershell
python ml_pipeline.py predict `
  --model .\artifacts\model.joblib `
  --data .\data\unlabeled.csv `
  --output .\artifacts\predictions.csv
```

Only load model artifacts from trusted sources: joblib artifacts use Python
pickle serialization and can execute code when loaded.

## Serve predictions over HTTP

The API loads the model at startup, provides `/health` and `/predict`, and
returns a validation error if required features are missing. It binds to
localhost by default; specify a host explicitly before exposing it on a network.

```powershell
python api.py --model .\artifacts\model.joblib
```

Send JSON records to `http://127.0.0.1:8000/predict`:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' `
  -Body '{"records":[{"study_hours":7.2,"attendance_pct":81,"practice_tests":4,"program":"accelerated"}]}'
```

The API also exposes `/metadata` to view the trained target and feature names, and each prediction includes a per-row `confidence` value when the model supports probability scores.

Example response:

```json
{
  "predictions": ["yes"],
  "count": 1,
  "details": [{"index": 0, "prediction": "yes", "confidence": 0.91}]
}
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Containerize with Docker

Build and run the API in a container:

```powershell
docker build -t end-to-end-ml-pipeline .
docker run --rm -p 8000:8000 end-to-end-ml-pipeline
```

The container exposes the same FastAPI app on port 8000 and loads the model from
`artifacts/model.joblib` by default. If you want to pass a different model path,
override the command when running the container.

## Streamlit dashboard

Run the dashboard for an interactive CSV upload and prediction UI:

```powershell
python -m streamlit run streamlit_app.py
```

The dashboard loads a saved model, shows metadata, and lets you upload a CSV to
preview predictions and download the results.

### Publish a live demo

The app is prepared for Streamlit Community Cloud and uses the committed demo
model at `artifacts/model.joblib`. To publish it, sign in to
[Streamlit Community Cloud](https://share.streamlit.io/), create a new app from
this GitHub repository, select the `main` branch and `streamlit_app.py`, then
deploy. Streamlit will install dependencies from `requirements.txt`. The demo
accepts uploaded feature CSVs; do not upload sensitive or personal data.

After deployment, replace this line with the public app URL:
`Live demo: not deployed yet`.

## Test

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

For a clean local verification, use a fresh Python 3.10+ environment and follow
the Quick start. GitHub Actions independently installs the development
dependencies, runs tests on Python 3.10 through 3.13, checks linting and
formatting, and builds the Docker image on pushes and pull requests to `main`.
