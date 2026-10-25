"""Independent inference check: load the bundle and predict the sample payload directly."""
from pathlib import Path
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
with (ROOT / "backend/test_payload.json").open(encoding="utf-8") as f:
    payload = json.load(f)

features = bundle["features"]
assert set(features) == set(payload), "Payload keys must match features"

frame = pd.DataFrame([[payload[f] for f in features]], columns=features)
model = bundle["pipeline"]
pred = model.predict(frame).tolist()
label = bundle["classes"][str(pred[0])]
print("Prediction (encoded):", pred)
print("Prediction (label):  ", label)
assert label in bundle["classes"].values()
print("PASS: local inference")
