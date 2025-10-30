# ==============================
# DistilBERT + MLP (pipeline complet pour MLflow)
# ==============================
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
import mlflow.sklearn
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle

# ==============================
# 1️⃣ Chargement du dataset
# ==============================
candidates = [Path("data/sentiment140_light.csv"), Path("../data/sentiment140_light.csv")]
data_path = next((p for p in candidates if p.exists()), None)
if data_path is None:
    raise FileNotFoundError("Impossible de trouver data/sentiment140_light.csv ou ../data/sentiment140_light.csv")

df = pd.read_csv(data_path, encoding="utf-8")
X = df["text"]
y = df["target"]

# ==============================
# 2️⃣ Split train / val / test
# ==============================
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)

# ==============================
# 3️⃣ Encapsuler DistilBERT comme transformer scikit-learn
# ==============================
class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="distilbert-base-uncased", batch_size=64, max_length=96):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def transform(self, X):
        embs = []
        n_batches = int(np.ceil(len(X) / self.batch_size))
        for i in tqdm(range(0, len(X), self.batch_size), total=n_batches, desc="Encodage BERT"):
            batch = list(X[i:i+self.batch_size])
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            mean_emb = outputs.last_hidden_state.mean(dim=1)
            embs.append(mean_emb.cpu().numpy())
        return np.vstack(embs)

    def fit(self, X, y=None):
        return self

# ==============================
# 4️⃣ Pipeline complet BERT + MLP
# ==============================
model = Pipeline([
    ("bert", BertVectorizer()),
    ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42))
])

# ==============================
# 5️⃣ Entraînement
# ==============================
model.fit(X_train, y_train)

# ==============================
# 6️⃣ Évaluation
# ==============================
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

val_acc  = accuracy_score(y_val, y_val_pred)
val_f1   = f1_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1  = f1_score(y_test, y_test_pred)

print("=== DistilBERT + MLP ===")
print(f"Accuracy (val): {val_acc:.4f}")
print(f"F1-score (val): {val_f1:.4f}")
print(f"Accuracy (test): {test_acc:.4f}")
print(f"F1-score (test): {test_f1:.4f}")

# ==============================
# 7️⃣ Logging MLflow (identique au TF-IDF)
# ==============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment140_Models")

with mlflow.start_run(run_name="DistilBERT_MLP"):
    mlflow.log_param("model", "DistilBERT_MLP")
    mlflow.log_param("embedding_model", "distilbert-base-uncased")
    mlflow.log_param("hidden_layers", "(64, 32)")
    mlflow.log_param("max_iter", 150)
    mlflow.log_param("framework", "PyTorch + scikit-learn")

    mlflow.log_metric("val_accuracy",  val_acc)
    mlflow.log_metric("val_f1",        val_f1)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1",       test_f1)

    mlflow.sklearn.log_model(model, "model")

print("\n✅ Modèle DistilBERT + MLP loggé dans MLflow avec succès")

# ==============================
# 8️⃣ Sauvegarde locale
# ==============================
Path("models").mkdir(exist_ok=True)
with open(Path("models") / "bert_mlp_pipeline.pkl", "wb") as f:
    pickle.dump(model, f)

print("📦 Sauvegardé dans models/bert_mlp_pipeline.pkl")
print(" Lancez `mlflow ui --backend-store-uri ./mlruns` pour le visualiser.")
