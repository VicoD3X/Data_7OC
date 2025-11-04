# main.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
import joblib
import numpy as np
import os, logging

# ----- Azure App Insights logger -----
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger("airparadis.api")
logger.setLevel(logging.INFO)
_conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if _conn:
    logger.addHandler(AzureLogHandler(connection_string=_conn))

def log_bad_pred(tweet_text: str, y_pred: int, y_proba: float | None, model_version: str = "tfidf-logreg-1.0"):
    logger.warning(
        "bad_pred",
        extra={
            "custom_dimensions": {
                "kind": "bad_pred",
                "tweet_text": tweet_text,
                "prediction": int(y_pred),
                "probability": float(y_proba) if y_proba is not None else None,
                "model_version": model_version,
            }
        },
    )

# ----- App + modèle -----
app = FastAPI()
MODEL_VERSION = os.getenv("MODEL_VERSION", "tfidf-logreg-1.0")
model = joblib.load("models/tfidf_vectorizer.pkl")  # pipeline (vectorizer + clf)

# ----- Schémas -----
class PredictOneIn(BaseModel):
    tweet: str

class PredictOneOut(BaseModel):
    prediction: int
    proba: float | None = None
    model_version: str

class FeedbackIn(BaseModel):
    tweet: str
    prediction: int
    proba: float | None = None
    reason: str | None = "user_marked_wrong"

@app.get("/")
def root():
    return {"message": "API is running", "model_version": MODEL_VERSION}

# Route existante (compat) : liste brute (ex: [["mon tweet"], ["autre tweet"]])
@app.post("/predict")
def predict(data: list = Body(...)):
    arr = np.array(data)
    preds = model.predict(arr).tolist()
    return {"predictions": preds, "model_version": MODEL_VERSION}

# Nouvelle route simple pour 1 tweet
@app.post("/predict_one", response_model=PredictOneOut)
def predict_one(payload: PredictOneIn):
    tweet = payload.tweet
    y_pred = int(model.predict([tweet])[0])
    # proba si dispo
    try:
        proba = float(model.predict_proba([tweet])[0].max())
    except Exception:
        proba = None
    return PredictOneOut(prediction=y_pred, proba=proba, model_version=MODEL_VERSION)

# Route feedback : le front envoie le tweet jugé mal prédit
@app.post("/feedback")
def feedback(payload: FeedbackIn):
    log_bad_pred(tweet_text=payload.tweet, y_pred=payload.prediction, y_proba=payload.proba, model_version=MODEL_VERSION)
    return {"ok": True}





# Pour lancer l'API :
# 1) activer l'environnement virtuel :
#    .venv\Scripts\Activate   (Windows)
#    source .venv/bin/activate (Linux/Mac)
# 2) démarrer le serveur :
#    uvicorn app.main:app --reload
# 3) ouvrir dans le navigateur :
#    http://127.0.0.1:8000
#    http://127.0.0.1:8000/docs pour tester /predict
