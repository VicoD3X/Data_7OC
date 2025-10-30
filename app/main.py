# app/main.py
from fastapi import FastAPI, Body
import joblib
import numpy as np
import requests
from pathlib import Path

# ============================================================
# ⚠️ ALIAS __main__ : expose la classe BertVectorizer sous le
#    nom attendu par le pickle (souvent "__main__.BertVectorizer")
#    si le modèle a été sérialisé depuis un notebook / script.
# ============================================================
import sys
from app.bert_vectorizer import BertVectorizer  # classe réelle
sys.modules['__main__'].BertVectorizer = BertVectorizer  # alias pour joblib

app = FastAPI()

# === Téléchargement auto du modèle BERT depuis Google Drive (si absent) ===
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "bert_mlp_pipeline.pkl"

FILE_ID = "1w1PMzHv5R2MAapQlllOQU0ezaT8O2ucm"  # ton ID Drive
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

MODEL_DIR.mkdir(exist_ok=True)
if not MODEL_PATH.exists():
    print("🔽 Téléchargement du modèle BERT depuis Google Drive…")
    r = requests.get(URL)
    r.raise_for_status()
    MODEL_PATH.write_bytes(r.content)
    print("✅ Modèle téléchargé !")

# === Chargement du pipeline picklé (BERTVectorizer + MLP) ===
print("⚙️ Chargement du modèle BERT…")
model = joblib.load(MODEL_PATH)  # joblib retrouve __main__.BertVectorizer
print("✅ Modèle BERT chargé avec succès !")

# === Endpoints FastAPI ===
@app.get("/")
def root():
    return {"message": "API is running with BERT model"}

@app.post("/predict")
def predict(data: list = Body(...)):
    arr = np.array(data)
    preds = model.predict(arr).tolist()
    return {"predictions": preds}

# === Lancement local (rappel) ===
# uvicorn app.main:app --workers 1
# http://127.0.0.1:8000  |  http://127.0.0.1:8000/docs







# === Instructions locales ===
# 1) activer l'environnement virtuel :
#    .venv\Scripts\activate   (Windows)
#    source .venv/bin/activate (Linux/Mac)
# 2) lancer le serveur :
#    uvicorn app.main:app --reload
# 3) ouvrir :
#    http://127.0.0.1:8000
#    http://127.0.0.1:8000/docs
