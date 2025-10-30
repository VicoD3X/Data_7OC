# app/streamlit_app.py

# --- rendre importable le paquet "app" même si Streamlit change le cwd ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # racine du repo (celle qui contient /app et /models)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pickle

# ============================================================
# ⚠️ ALIAS __main__ : même logique que côté API pour que pickle
#    retrouve la classe si le modèle a été sérialisé depuis __main__.
# ============================================================
from app.bert_vectorizer import BertVectorizer
sys.modules['__main__'].BertVectorizer = BertVectorizer  # alias pour pickle

@st.cache_resource
def load_model():
    # On pointe vers le modèle BERT (et non plus TF-IDF)
    model_path = (ROOT / "models" / "bert_mlp_pipeline.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)  # retrouve __main__.BertVectorizer grâce à l'alias
    return model

# Chargement une seule fois
model = load_model()

# === Interface ===
st.title("Analyse de sentiment - Air Paradis (BERT ✈️)")

tweet = st.text_area(
    "Saisissez un tweet",
    height=120,
    placeholder="Exemple : Flight delayed again..."
)

if st.button("Prédire"):
    if not tweet.strip():
        st.warning("Veuillez saisir un tweet.")
    else:
        pred = int(model.predict([tweet])[0])
        label = "Positif" if pred == 1 else "Négatif"
        st.write(f"Sentiment prédit : {label}")

        # Affichage des probabilités si dispo
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([tweet])[0]
            st.write(f"Probabilité négatif : {proba[0]:.3f} | probabilité positif : {proba[1]:.3f}")

# === Lancer en local ===
# 1) .\occ-env\Scripts\activate
# 2) streamlit run app/streamlit_app.py
