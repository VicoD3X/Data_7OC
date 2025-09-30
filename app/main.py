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
