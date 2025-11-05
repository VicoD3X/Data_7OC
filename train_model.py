import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import mlflow
import mlflow.sklearn
import pickle
import re
import spacy

# ==============================
# 1️⃣ Chargement des données
# ==============================
df = pd.read_csv("data/sentiment140_light.csv")
X = df["text"]
y = df["target"]

# ==============================
# Prétraitement : lemmatisation
# ==============================
print("Prétraitement : lemmatisation et nettoyage du texte...")

# Charger le modèle linguistique spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # enlève ponctuation / chiffres
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

X = X.apply(preprocess_text)

# ==============================
# 2️⃣ Split train / val / test (≈ 70 / 15 / 15)
# ==============================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
)

# ==============================
# 3️⃣ Pipeline TF-IDF + Logistic Regression
# ==============================
model = make_pipeline(
    TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    ),
    LogisticRegression(max_iter=500)
)

model.fit(X_train, y_train)

# ==============================
# 4️⃣ Évaluation
# ==============================
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

val_acc = accuracy_score(y_val, y_val_pred)
val_f1  = f1_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1  = f1_score(y_test, y_test_pred)

print("=== TF-IDF + Logistic Regression ===")
print("=== Validation ===")
print(classification_report(y_val, y_val_pred))
print(f"Accuracy (val) : {val_acc:.4f}")
print(f"F1-score (val) : {val_f1:.4f}")

print("\n=== Test ===")
print(classification_report(y_test, y_test_pred))
print(f"Accuracy (test) : {test_acc:.4f}")
print(f"F1-score (test) : {test_f1:.4f}")

# ==============================
# 5️⃣ MLflow logging
# ==============================
mlflow.set_experiment("Sentiment140_Models")

with mlflow.start_run(run_name="TFIDF_LogReg"):
    mlflow.log_param("model", "tfidf_logistic_regression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "(1, 2)")
    mlflow.log_param("max_iter", 500)
    mlflow.log_param("preprocessing", "lemmatization (spaCy)")

    mlflow.log_metric("val_accuracy",  val_acc)
    mlflow.log_metric("val_f1",        val_f1)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1",       test_f1)

    # Sauvegarde MLflow
    mlflow.sklearn.log_model(model, "model")

# ==============================
# 6️⃣ Sauvegarde du modèle pour Streamlit
# ==============================
Path("models").mkdir(exist_ok=True)
with open(Path("models") / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Modèle TF-IDF sauvegardé dans models/tfidf_vectorizer.pkl")
print(" Lancez `mlflow ui --backend-store-uri ./mlruns` pour visualiser les résultats.")
