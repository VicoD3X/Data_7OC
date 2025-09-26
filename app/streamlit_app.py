import streamlit as st
import pickle
from pathlib import Path

# =======================
# Charger le modèle sauvegardé
# =======================
@st.cache_resource
def load_model():
    model_path = Path("models/trained_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# =======================
# Interface Streamlit
# =======================
st.title("Démo classification - Tutoriel")

st.write("Exemple de prédiction (jeu de données Iris).")

# Dans le tuto : 4 valeurs numériques pour les fleurs Iris
sepal_length = st.number_input("Sepal length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width", 0.0, 10.0, 0.2)

if st.button("Prédire"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    st.success(f"Résultat : classe prédite = {prediction}")
