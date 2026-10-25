import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app import app, FEATURES  # noqa: E402

PAYLOAD = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))


def test_health():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["features"] == len(FEATURES)


def test_predict_valid():
    response = app.test_client().post("/predict", json=PAYLOAD)
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["prediction"] in {"healthy", "diseased"}
    assert set(body["votes"]) == {"naive_bayes", "random_forest", "svc"}


def test_predict_missing_feature():
    incomplete = dict(PAYLOAD)
    incomplete.pop(FEATURES[0])
    response = app.test_client().post("/predict", json=incomplete)
    assert response.status_code == 400
    assert FEATURES[0] in response.get_json()["missing"]


def test_predict_extra_feature():
    response = app.test_client().post("/predict", json={**PAYLOAD, "target": "healthy"})
    assert response.status_code == 400
    assert "target" in response.get_json()["extra"]


def test_predict_bad_values():
    bad = {**PAYLOAD, "age": "old", "gender": "Robot"}
    response = app.test_client().post("/predict", json=bad)
    assert response.status_code == 400
    assert len(response.get_json()["details"]) == 2


def test_predict_invalid_json():
    response = app.test_client().post("/predict", data="not-json",
                                      content_type="application/json")
    assert response.status_code == 400
