# app/streamlit_app.py
import streamlit as st
import pickle
from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path("models/tfidf_vectorizer.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Analyse de sentiment - Air Paradis")

tweet = st.text_area("Saisissez un tweet", height=120, placeholder="Exemple : Flight delayed again...")

if st.button("Prédire"):
    if not tweet.strip():
        st.warning("Veuillez saisir un tweet.")
    else:
        pred = int(model.predict([tweet])[0])
        label = "Positif" if pred == 1 else "Négatif"

        st.write(f"Sentiment prédit : {label}")

        # Affichage optionnel des probabilités si disponible
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([tweet])[0]
            st.write(f"Probabilité négatif : {proba[0]:.3f} | probabilité positif : {proba[1]:.3f}")



# =======================
# COMMENT ACTIVER LE MODE DÉVELOPPEUR STREAMLIT
# Dans le terminal, taper : .\.venv\Scripts\Activate
# Puis : streamlit run app/streamlit_app.py
# Pour forcer le rechargement automatique du script à chaque sauvegarde : tapper "R" dans le terminal
# Pour forcer le rechargement du script (sans attendre la sauvegarde) : tapper "D" dans le terminal
# Pour arrêter le serveur Streamlit : tapper "CTRL + C" dans le terminal
# Pour plus d'info : https://docs.streamlit.io/library/get-started/installation
# =======================