from fastapi.testclient import TestClient
from app.main import app

def test_root_ok():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json().get("message") == "API is running"

def test_predict_ok():
    client = TestClient(app)
    payload = ["great flight", "horrible service"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    preds = r.json().get("predictions")
    assert isinstance(preds, list) and len(preds) == 2
    for p in preds:
        assert p in [0, 1]
