"""Minimal Flask API for the lifestyle health-risk ensemble.

GET  /health   -> service status and model info
POST /predict  -> JSON object with all model features -> prediction + per-model votes
Run from the project root:  python backend/app.py
"""
from pathlib import Path
import math

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models/model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]
SCHEMA = BUNDLE["schema"]
CLASSES = BUNDLE["classes"]
TASK = BUNDLE.get("task", "classification")

app = Flask(__name__)


def validate(data: dict) -> list[str]:
    """Check every value against the saved input schema. Returns a list of problems."""
    problems = []
    for name in FEATURES:
        rule, value = SCHEMA[name], data[name]
        if rule["kind"] == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                problems.append(f"{name}: must be a number")
            elif not rule["min"] <= value <= rule["max"]:
                problems.append(f"{name}: must be between {rule['min']} and {rule['max']}")
        elif value not in rule["options"]:
            problems.append(f"{name}: must be one of {rule['options']}")
    return problems


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK,
                   model_version=BUNDLE.get("model_version"),
                   methodology=BUNDLE.get("methodology"))


@app.get("/schema")
def schema():
    return jsonify(features=FEATURES, schema=SCHEMA)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400

    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch", missing=missing, extra=extra), 400

    problems = validate(data)
    if problems:
        return jsonify(error="Invalid input values", details=problems), 400

    frame = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    try:
        encoded = int(MODEL.predict(frame)[0])
        # Individual votes of the three base learners (majority vote = final answer)
        transformed = MODEL.named_steps["preprocess"].transform(frame)
        ensemble = MODEL.named_steps["ensemble"]
        votes = {name: CLASSES[str(int(ensemble.le_.classes_[est.predict(transformed)[0]]))]
                 for name, est in ensemble.named_estimators_.items()}
        label = CLASSES[str(encoded)]
        return jsonify(
            prediction=label,
            votes=votes,
            agreement=f"{sum(v == label for v in votes.values())} of {len(votes)} models",
            task=TASK,
            disclaimer="Research/teaching prototype only - not a medical diagnosis.",
        )
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
