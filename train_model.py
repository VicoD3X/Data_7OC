# train_model.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import pickle
import numpy as np
from gensim.models import Word2Vec

# 1) Chargement du dataset
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

# 3) Entraînement du modèle Word2Vec
sentences = [text.split() for text in X_train]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

# Fonction pour vectoriser les textes
def vectorize_texts(texts, model, vector_size=100):
    vectors = []
    for text in texts:
        words = text.split()
        word_vecs = [model.wv[word] for word in words if word in model.wv]
        if len(word_vecs) > 0:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(vector_size))
    return np.array(vectors)

X_train_vec = vectorize_texts(X_train, w2v_model)
X_val_vec   = vectorize_texts(X_val, w2v_model)
X_test_vec  = vectorize_texts(X_test, w2v_model)

# 4) Entraînement Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

# 5) Évaluation
y_val_pred  = model.predict(X_val_vec)
y_test_pred = model.predict(X_test_vec)

val_acc  = accuracy_score(y_val, y_val_pred)
val_f1   = f1_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1  = f1_score(y_test, y_test_pred)

print("=== Word2Vec + Logistic Regression ===")
print("=== Validation ===")
print(classification_report(y_val, y_val_pred))
print(f"Accuracy (val) : {val_acc:.4f}")
print(f"F1-score (val) : {val_f1:.4f}")

print("\n=== Test ===")
print(classification_report(y_test, y_test_pred))
print(f"Accuracy (test) : {test_acc:.4f}")
print(f"F1-score (test) : {test_f1:.4f}")

# 6) MLflow logging
mlflow.set_experiment("Sentiment140_Models")

with mlflow.start_run(run_name="Word2Vec_LogReg"):
    mlflow.log_param("model", "word2vec_logistic_regression")
    mlflow.log_param("vector_size", 100)
    mlflow.log_param("window", 5)
    mlflow.log_param("min_count", 2)
    mlflow.log_param("max_iter", 500)

    mlflow.log_metric("val_accuracy",  val_acc)
    mlflow.log_metric("val_f1",        val_f1)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1",       test_f1)

    # Sauvegarde du modèle W2V et du classifieur
    Path("models").mkdir(exist_ok=True)
    w2v_model.save("models/word2vec_model.bin")
    mlflow.log_artifact("models/word2vec_model.bin", artifact_path="embeddings")
    mlflow.sklearn.log_model(model, "model")

# 7) Sauvegarde pour Streamlit (wrapper)
class Word2VecLogRegWrapper:
    def __init__(self, w2v_model, clf, vector_size=100):
        self.w2v_model = w2v_model
        self.clf = clf
        self.vector_size = vector_size

    def vectorize(self, texts):
        vectors = []
        for text in texts:
            words = text.split()
            word_vecs = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            if len(word_vecs) > 0:
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)

    def predict(self, texts):
        vecs = self.vectorize(texts)
        return self.clf.predict(vecs)

# Sauvegarde du wrapper complet
wrapper = Word2VecLogRegWrapper(w2v_model, model)
with open(Path("models") / "trained_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
    
print("\n Modèle FastText sauvegardé dans models/trained_model.pkl")
print(" Lancez `mlflow ui --backend-store-uri ./mlruns` pour visualiser les résultats.")


# Pour exécuter ce script :
# activer l'environnement virtuel : .\.venv\Scripts\Activate
# pour exécuter le script : python train_model.py
# puis lancer l'application Streamlit : streamlit run app/streamlit_app.py