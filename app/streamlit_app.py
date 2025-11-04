# app/streamlit_app.py
import os
import requests
import streamlit as st
import pickle
from pathlib import Path

st.set_page_config(page_title="Analyse de sentiment - Air Paradis", page_icon="✈️")

# -----------------------
# Configuration
# -----------------------
# ⚙️ Définis l’URL de ton API FastAPI dans .streamlit/secrets.toml :
# API_URL = "https://<TON-APP>.herokuapp.com"
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "")).strip()

# Option: activer un fallback local si l’API est indisponible
USE_LOCAL_FALLBACK = st.secrets.get("USE_LOCAL_FALLBACK", True)

# -----------------------
# Modèle local (fallback)
# -----------------------
@st.cache_resource
def load_local_model():
    model_path = Path("models/tfidf_vectorizer.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

local_model = load_local_model()

def predict_local(tweet: str):
    if not local_model:
        raise RuntimeError("Modèle local introuvable (models/tfidf_vectorizer.pkl).")
    pred = int(local_model.predict([tweet])[0])
    proba = None
    if hasattr(local_model, "predict_proba"):
        # max des deux classes (0/1) comme confiance
        proba = float(local_model.predict_proba([tweet])[0].max())
    return pred, proba

# -----------------------
# Appels API
# -----------------------
def predict_api(tweet: str):
    if not API_URL:
        raise RuntimeError("API_URL non configuré.")
    url = f"{API_URL}/predict_one"
    r = requests.post(url, json={"tweet": tweet}, timeout=15)
    r.raise_for_status()
    data = r.json()
    pred = int(data["prediction"])
    proba = data.get("proba", None)
    if proba is not None:
        proba = float(proba)
    return pred, proba

def send_feedback_api(tweet: str, pred: int, proba: float | None):
    if not API_URL:
        raise RuntimeError("API_URL non configuré.")
    url = f"{API_URL}/feedback"
    payload = {"tweet": tweet, "prediction": pred, "proba": proba, "reason": "user_marked_wrong"}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return True

# -----------------------
# UI
# -----------------------
st.title("Analyse de sentiment - Air Paradis")

with st.expander("⚙️ Configuration", expanded=False):
    st.write("**API_URL** :", API_URL if API_URL else "_(non défini)_")
    if not API_URL:
        st.info("Définis `API_URL` dans `.streamlit/secrets.toml` pour activer le logging Azure via l’API.")
    st.write("**Fallback local** :", "activé" if USE_LOCAL_FALLBACK else "désactivé")

tweet = st.text_area(
    "Saisissez un tweet",
    height=120,
    placeholder="Exemple : Flight delayed again, luggage lost…"
)

# état pour le feedback
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None
if "last_tweet" not in st.session_state:
    st.session_state.last_tweet = None
if "used_api" not in st.session_state:
    st.session_state.used_api = False

col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("Prédire", type="primary")
with col2:
    feedback_clicked = st.button("Signaler comme incorrect", disabled=st.session_state.last_pred is None)

if predict_clicked:
    if not tweet.strip():
        st.warning("Veuillez saisir un tweet.")
    else:
        try:
            # 1) on tente l’API (pour que le feedback déclenche le log Azure côté serveur)
            pred, proba = predict_api(tweet)
            st.session_state.used_api = True
        except Exception as api_err:
            # 2) fallback local si activé
            if USE_LOCAL_FALLBACK:
                try:
                    pred, proba = predict_local(tweet)
                    st.session_state.used_api = False
                    st.info("API indisponible → prédiction locale (pas de logging Azure).")
                except Exception as local_err:
                    st.error(f"Erreur de prédiction (API et local) : {local_err}")
                    pred, proba = None, None
            else:
                st.error(f"Erreur d'appel API : {api_err}")
                pred, proba = None, None

        if pred is not None:
            label = "Positif" if pred == 1 else "Négatif"
            st.success(f"Sentiment prédit : **{label}**")
            if proba is not None:
                st.caption(f"Confiance (classe la plus probable) : {proba:.3f}")

            # mémoriser pour feedback
            st.session_state.last_pred = pred
            st.session_state.last_proba = proba
            st.session_state.last_tweet = tweet

if feedback_clicked:
    if st.session_state.last_pred is None or not st.session_state.last_tweet:
        st.warning("Faites d’abord une prédiction.")
    else:
        if not st.session_state.used_api:
            st.info("Le feedback nécessite l’API pour être loggué dans Azure (pas disponible en mode local).")
        else:
            try:
                ok = send_feedback_api(
                    st.session_state.last_tweet,
                    st.session_state.last_pred,
                    st.session_state.last_proba
                )
                if ok:
                    st.success("Merci, votre signalement a été enregistré.")
            except Exception as e:
                st.error(f"Échec de l'envoi du signalement : {e}")





# =======================
# Raccourcis développeur
# streamlit run app/streamlit_app.py
# Arrêt : CTRL + C
# Docs : https://docs.streamlit.io/
# =======================



# =======================
# COMMENT ACTIVER LE MODE DÉVELOPPEUR STREAMLIT
# Dans le terminal, taper : .\.venv\Scripts\Activate
# Puis : streamlit run app/streamlit_app.py
# Pour forcer le rechargement automatique du script à chaque sauvegarde : tapper "R" dans le terminal
# Pour forcer le rechargement du script (sans attendre la sauvegarde) : tapper "D" dans le terminal
# Pour arrêter le serveur Streamlit : tapper "CTRL + C" dans le terminal
# Pour plus d'info : https://docs.streamlit.io/library/get-started/installation
# =======================