from fastapi import FastAPI, Body
import joblib
import numpy as np

app = FastAPI()

# Charger le modèle
model = joblib.load("models/trained_model.pkl")

@app.get("/")
def root():
    return {"message": "API is running"}

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
