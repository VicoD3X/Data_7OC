# train_model.py
import pickle
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow
import mlflow.sklearn

# 1) Chargement des données (chemins tolérants au répertoire courant)
candidates = [Path("data/sentiment140_light.csv"), Path("../data/sentiment140_light.csv")]
data_path = next((p for p in candidates if p.exists()), None)
if data_path is None:
    raise FileNotFoundError("Impossible de trouver data/sentiment140_light.csv ou ../data/sentiment140_light.csv")

df = pd.read_csv(data_path, encoding="utf-8")
X = df["text"]
y = df["target"]

# 2) Split train / val / test (≈ 70 / 15 / 15)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
)

# 3) Pipeline TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=300))
])

# 4) Entraînement
pipeline.fit(X_train, y_train)

# 5) Évaluations
y_val_pred  = pipeline.predict(X_val)
y_test_pred = pipeline.predict(X_test)

val_acc  = accuracy_score(y_val, y_val_pred)
val_f1   = f1_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1  = f1_score(y_test, y_test_pred)

print("=== Rapport Validation ===")
print(classification_report(y_val, y_val_pred))
print(f"Accuracy (val) : {val_acc:.4f}")
print(f"F1-score (val) : {val_f1:.4f}")

print("\n=== Rapport Test ===")
print(classification_report(y_test, y_test_pred))
print(f"Accuracy (test) : {test_acc:.4f}")
print(f"F1-score (test) : {test_f1:.4f}")

# 6) MLflow (une fois le modèle entraîné)
mlflow.set_experiment("Sentiment140_Models")
with mlflow.start_run(run_name="TFIDF_LogReg_final"):
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("stop_words", "english")
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("max_iter", 300)

    mlflow.log_metric("val_accuracy",  val_acc)
    mlflow.log_metric("val_f1",        val_f1)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1",       test_f1)

    # Log du pipeline complet (vectoriseur + modèle)
    mlflow.sklearn.log_model(pipeline, "model")

# 7) Sauvegarde du pipeline pour Streamlit / API
Path("models").mkdir(exist_ok=True)
with open(Path("models") / "trained_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModèle sauvegardé dans models/trained_model.pkl")
print("Pour visualiser MLflow : exécuter `mlflow ui` et ouvrir http://127.0.0.1:5000")



# Pour exécuter ce script :
# activer l'environnement virtuel : .\.venv\Scripts\Activate
# pour exécuter le script : python train_model.py
# puis lancer l'application Streamlit : streamlit run app/streamlit_app.py