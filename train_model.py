import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path

# =======================
# Jeu de données jouet
# =======================
data = {
    "tweet": [
        "J'adore voyager avec Air Paradis",
        "Air Paradis est nul, jamais à l'heure",
        "Super service et bons prix",
        "Horrible expérience, bagages perdus",
        "Vol agréable et staff sympa",
        "Très mauvaise compagnie aérienne"
    ],
    "label": [0, 1, 0, 1, 0, 1]  # 0 = positif/neutre, 1 = négatif
}

df = pd.DataFrame(data)

# =======================
# Pipeline (vectorizer + logistic regression)
# =======================
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", LogisticRegression())
])

# Entraînement
model.fit(df["tweet"], df["label"])

# =======================
# Sauvegarde du modèle
# =======================
Path("models").mkdir(exist_ok=True)
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modèle entraîné et sauvegardé dans models/trained_model.pkl")
