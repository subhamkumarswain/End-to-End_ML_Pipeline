"""Local HTTP API for serving a trained classification pipeline."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ml_pipeline import predict_frame


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)


def _build_prediction_details(
    bundle: dict[str, Any], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    model = bundle["model"]
    predictions = frame["prediction"]
    details: list[dict[str, Any]] = []
    if not hasattr(model, "predict_proba"):
        for index, prediction in enumerate(predictions):
            details.append({"index": index, "prediction": prediction, "confidence": None})
        return details

    features = bundle["features"]
    probabilities = model.predict_proba(frame[features])
    class_labels = list(model.classes_)
    for index, prediction in enumerate(predictions):
        try:
            confidence = float(probabilities[index][class_labels.index(prediction)])
        except (ValueError, TypeError):
            confidence = None
        details.append({"index": index, "prediction": prediction, "confidence": confidence})
    return details


def create_app(model_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict) or not {
            "model",
            "features",
            "target",
        }.issubset(bundle):
            raise RuntimeError(f"Model artifact has an invalid format: {model_path}")
        app.state.model_bundle = bundle
        yield

    app = FastAPI(
        title="Tabular Classification API",
        description="Predict target classes using a trained CSV classification model.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        if not hasattr(request.app.state, "model_bundle"):
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata(request: Request) -> dict[str, Any]:
        if not hasattr(request.app.state, "model_bundle"):
            raise HTTPException(status_code=503, detail="Model is not loaded")
        bundle = request.app.state.model_bundle
        model = bundle["model"]
        return {
            "target": bundle["target"],
            "features": bundle["features"],
            "model_type": type(model).__name__,
            "model_name": bundle.get("model_name", type(model).__name__),
            "model_version": bundle.get("model_version"),
            "classes": list(getattr(model, "classes_", [])),
        }

    @app.post("/predict")
    def predict_records(
        payload: PredictionRequest, request: Request
    ) -> dict[str, Any]:
        try:
            result = predict_frame(
                request.app.state.model_bundle,
                pd.DataFrame(payload.records),
            )
            predictions = result["prediction"].tolist()
            details = _build_prediction_details(request.app.state.model_bundle, result)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "predictions": predictions,
            "count": len(predictions),
            "details": details,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
