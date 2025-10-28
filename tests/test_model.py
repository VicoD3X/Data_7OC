import os
import pickle
import numpy as np

def test_model_file_exists():
    assert os.path.exists("models/tfidf_vectorizer.pkl"), "models/tfidf_vectorizer.pkl est manquant"

def test_model_prediction():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        model = pickle.load(f)
    X = ["I love this airline", "I hate delays"]
    y_pred = model.predict(X)
    assert len(y_pred) == 2
    assert set(np.unique(y_pred)).issubset({0, 1})
