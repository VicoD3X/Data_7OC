# app/streamlit_app.py
import os
import logging
from pathlib import Path
import pickle
import streamlit as st
from opencensus.ext.azure.log_exporter import AzureLogHandler

# -----------------------
# Azure App Insights (minimal)
# -----------------------
# => Version idempotente : évite l’empilement de handlers à chaque rerun Streamlit
@st.cache_resource
def get_logger():
    log = logging.getLogger("airparadis.streamlit")
    log.setLevel(logging.INFO)
    log.propagate = False  # évite la remontée vers le root logger (double logs)
    _conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")  # Heroku Config Var
    if _conn and not any(isinstance(h, AzureLogHandler) for h in log.handlers):
        handler = AzureLogHandler(connection_string=_conn)
        handler.setLevel(logging.INFO)
        log.addHandler(handler)
    return log

logger = get_logger()


def log_bad_pred(tweet_text: str, y_pred: int, y_proba: float | None = None):
    """Envoie un log 'bad_pred' avec le tweet + la prédiction (+ proba si dispo)"""
    dims = {
        "kind": "bad_pred",
        "source": "streamlit",
        "tweet_text": tweet_text,
        "prediction": int(y_pred),
    }
    if y_proba is not None:
        dims["probability"] = float(y_proba)
    logger.warning("bad_pred", extra={"custom_dimensions": dims})


def log_good_pred(tweet_text: str, y_pred: int, y_proba: float | None = None):
    """Envoie un log 'good_pred' avec le tweet + la prédiction (+ proba si dispo)"""
    dims = {
        "kind": "good_pred",
        "source": "streamlit",
        "tweet_text": tweet_text,
        "prediction": int(y_pred),
    }
    if y_proba is not None:
        dims["probability"] = float(y_proba)
    logger.warning("good_pred", extra={"custom_dimensions": dims})  # même niveau que bad_pred

# -----------------------
# App Streamlit
# -----------------------
st.set_page_config(page_title="Analyse de sentiment - Air Paradis", page_icon="✈️")

@st.cache_resource
def load_model():
    model_path = Path("models/tfidf_vectorizer.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Analyse de sentiment - Air Paradis")

tweet = st.text_area("Saisissez un tweet", height=120, placeholder="Exemple : Flight delayed again...")

# État pour pouvoir cliquer "Signaler" après "Prédire"
if "last_tweet" not in st.session_state:
    st.session_state.last_tweet = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None

col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("Prédire", type="primary", key="btn_predict")

# --- Prédiction ---
if predict_clicked:
    if not tweet.strip():
        st.warning("Veuillez saisir un tweet.")
    else:
        pred = int(model.predict([tweet])[0])
        label = "Positif" if pred == 1 else "Négatif"

        # ✅ Affichage simplifié pour la version production
        st.success(f"Sentiment détecté : {label}")

        proba_max = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([tweet])[0]
            proba_max = float(max(proba[0], proba[1]))  # Conservée pour les logs, non affichée

        # 🔁 Mémorisation pour le feedback utilisateur
        st.session_state.last_tweet = tweet
        st.session_state.last_pred = pred
        st.session_state.last_proba = proba_max

        # 🪵 Log automatique vers Azure Insights
        try:
            log_good_pred(tweet, pred, proba_max)
        except Exception as e:
            st.warning(f"Échec du log Azure Insights : {e}")


# --- Bouton "Signaler" rendu APRÈS la mise à jour de l'état ---
with col2:
    if st.session_state.last_pred is not None:
        if st.button("Signaler comme incorrect", key="btn_feedback"):
            try:
                log_bad_pred(
                    st.session_state.last_tweet,
                    st.session_state.last_pred,
                    st.session_state.last_proba
                )
                st.success("Merci, votre signalement a été enregistré.")
            except Exception as e:
                st.error(f"Impossible d'envoyer le signalement: {e})")






# =======================
# COMMENT ACTIVER LE MODE DÉVELOPPEUR STREAMLIT
# Dans le terminal, taper : .\.venv\Scripts\Activate
# Puis : streamlit run app/streamlit_app.py
# Pour forcer le rechargement automatique du script à chaque sauvegarde : tapper "R" dans le terminal
# Pour forcer le rechargement du script (sans attendre la sauvegarde) : tapper "D" dans le terminal
# Pour arrêter le serveur Streamlit : tapper "CTRL + C" dans le terminal
# Pour plus d'info : https://docs.streamlit.io/library/get-started/installation
# =======================
