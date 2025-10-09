import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from fastapi.testclient import TestClient


client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "API is running" in res.json()["message"]

def test_predict():
    payload = [[5.1, 3.5, 1.4, 0.2]]
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "predictions" in res.json()




# lancement  # pytest tests/test_api.py -v
# pytest -vv
