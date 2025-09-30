import pickle
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Charger le dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("✅ Modèle Iris entraîné avec LogisticRegression")

# Sauvegarde du modèle
Path("models").mkdir(exist_ok=True)
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("📂 Modèle sauvegardé dans models/trained_model.pkl")



# Pour exécuter ce script :
# activer l'environnement virtuel : .\.venv\Scripts\Activate
# pour exécuter le script : python train_model.py
# puis lancer l'application Streamlit : streamlit run app/streamlit_app.py