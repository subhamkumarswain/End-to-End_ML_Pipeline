# End-to-end tabular classification pipeline

This project validates CSV inputs, tunes a scikit-learn classifier with
cross-validation, evaluates it on a stratified holdout set, saves the fitted
preprocessing and model together, records experiment metrics, and predicts from
the CLI or a local HTTP API.

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

The comparison saves metrics for each candidate model and keeps the one with the
highest macro F1 score. When a version is supplied, the project also writes a
versioned artifact copy such as `best_model-v1.joblib`, while preserving the
latest model at the regular path.

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

## Test

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
