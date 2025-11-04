# app/streamlit_app.py
import os
import logging
from pathlib import Path
import pickle
import streamlit as st

# -----------------------
# Azure App Insights (minimal)
# -----------------------
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger("airparadis.streamlit")
logger.setLevel(logging.INFO)
_conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")  # vient de Heroku Config Vars
if _conn:
    logger.addHandler(AzureLogHandler(connection_string=_conn))

def log_bad_pred(tweet_text: str, y_pred: int, y_proba: float | None = None):
    # Envoie un log "bad_pred" avec le tweet + la prédiction (+ proba si dispo)
    dims = {
        "kind": "bad_pred",
        "source": "streamlit",
        "tweet_text": tweet_text,
        "prediction": int(y_pred),
    }
    if y_proba is not None:
        dims["probability"] = float(y_proba)
    logger.warning("bad_pred", extra={"custom_dimensions": dims})

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

# Boutons
col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("Prédire", type="primary")
with col2:
    feedback_clicked = st.button("Signaler comme incorrect", disabled=st.session_state.last_pred is None)

if predict_clicked:
    if not tweet.strip():
        st.warning("Veuillez saisir un tweet.")
    else:
        pred = int(model.predict([tweet])[0])
        label = "Positif" if pred == 1 else "Négatif"
        st.write(f"Sentiment prédit : {label}")

        proba_max = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([tweet])[0]
            st.write(f"Probabilité négatif : {proba[0]:.3f} | probabilité positif : {proba[1]:.3f}")
            proba_max = float(max(proba[0], proba[1]))

        # Mémoriser pour pouvoir signaler ensuite
        st.session_state.last_tweet = tweet
        st.session_state.last_pred = pred
        st.session_state.last_proba = proba_max

if feedback_clicked:
    # Envoie le log à Azure (si la connection string est présente)
    if st.session_state.last_tweet and st.session_state.last_pred is not None:
        try:
            log_bad_pred(st.session_state.last_tweet, st.session_state.last_pred, st.session_state.last_proba)
            st.success("Merci, votre signalement a été enregistré.")
        except Exception as e:
            st.error(f"Impossible d'envoyer le signalement: {e}")
    else:
        st.warning("Faites d’abord une prédiction.")

# =======================
# COMMENT ACTIVER LE MODE DÉVELOPPEUR STREAMLIT
# Dans le terminal, taper : .\.venv\Scripts\Activate
# Puis : streamlit run app/streamlit_app.py
# Pour forcer le rechargement automatique du script à chaque sauvegarde : tapper "R" dans le terminal
# Pour forcer le rechargement du script (sans attendre la sauvegarde) : tapper "D" dans le terminal
# Pour arrêter le serveur Streamlit : tapper "CTRL + C" dans le terminal
# Pour plus d'info : https://docs.streamlit.io/library/get-started/installation
# =======================
