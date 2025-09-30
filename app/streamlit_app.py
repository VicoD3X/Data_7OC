# app/streamlit_app.py
import streamlit as st
import pickle
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris

# =======================
# Charger le modèle
# =======================
@st.cache_resource
def load_model():
    model_path = Path("models/trained_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Charger noms des classes (Iris setosa, versicolor, virginica)
iris = load_iris()
class_names = iris.target_names

# =======================
# Interface Streamlit
# =======================
st.title("🌸 Démo classification Iris")

st.write("Entrez les caractéristiques de la fleur :")

# Inputs utilisateur
sepal_length = st.number_input("Sepal length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width", 0.0, 10.0, 0.2)

if st.button("Prédire"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"Résultat : classe prédite = {class_names[prediction]}")



# =======================
# COMMENT ACTIVER LE MODE DÉVELOPPEUR STREAMLIT
# Dans le terminal, taper : .\.venv\Scripts\Activate
# Puis : streamlit run app/streamlit_app.py
# Pour forcer le rechargement automatique du script à chaque sauvegarde : tapper "R" dans le terminal
# Pour forcer le rechargement du script (sans attendre la sauvegarde) : tapper "D" dans le terminal
# Pour arrêter le serveur Streamlit : tapper "CTRL + C" dans le terminal
# Pour plus d'info : https://docs.streamlit.io/library/get-started/installation
# =======================