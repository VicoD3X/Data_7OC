from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path

# Charger le modèle
model_path = Path("models/trained_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Créer l'app FastAPI
app = FastAPI(title="API Air Paradis - Sentiment Analysis")

# Définir la structure de la requête
class TweetInput(BaseModel):
    text: str

# Route de test
@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API Air Paradis"}

# Route de prédiction
@app.post("/predict")
def predict_sentiment(input_data: TweetInput):
    prediction = model.predict([input_data.text])[0]
    sentiment = "Negatif" if prediction == 1 else "Positif/Neutre"
    return {"tweet": input_data.text, "prediction": int(prediction), "sentiment": sentiment}
