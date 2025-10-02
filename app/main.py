# app/main.py
from fastapi import FastAPI, Body
import pickle
import numpy as np
from pathlib import Path

# Charger le modèle sauvegardé
model_path = Path("models/trained_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="API Iris - Démo")

@app.get("/")
def root():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: list = Body(...)):
    arr = np.array(data)
    preds = model.predict(arr).tolist()
    return {"predictions": preds}








# Pour lancer l'API :
# 1) activer l'environnement virtuel :
#    .venv\Scripts\Activate   (Windows)
#    source .venv/bin/activate (Linux/Mac)
# 2) démarrer le serveur :
#    uvicorn app.main:app --reload
# 3) ouvrir dans le navigateur :
#    http://127.0.0.1:8000
#    http://127.0.0.1:8000/docs pour tester /predict
