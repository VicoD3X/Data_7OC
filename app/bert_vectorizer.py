# app/bert_vectorizer.py

# === Import des librairies nécessaires ===
# numpy : manipulation des matrices
# torch : moteur de calcul pour PyTorch
# tqdm : affichage d'une barre de progression lors du traitement
# sklearn.base : pour rendre la classe compatible avec un pipeline scikit-learn
# transformers : pour charger le modèle BERT et son tokenizer
import numpy as np
import torch
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, AutoModel


# === Classe personnalisée compatible avec scikit-learn ===
class BertVectorizer(BaseEstimator, TransformerMixin):
    """
    Cette classe transforme une liste de textes en vecteurs numériques (embeddings)
    à l’aide de DistilBERT, une version légère de BERT.
    
    Elle se comporte comme un transformer scikit-learn classique :
    - fit() : pour compatibilité (ne fait rien ici)
    - transform() : convertit les textes en vecteurs
    """

    def __init__(self, model_name="distilbert-base-uncased", batch_size=64, max_length=96):
        # Nom du modèle BERT à utiliser (ici, DistilBERT pour plus de légèreté)
        self.model_name = model_name

        # Nombre de textes traités par lot (batch) pour éviter la surcharge mémoire
        self.batch_size = batch_size

        # Longueur maximale des séquences de tokens (les tweets sont courts)
        self.max_length = max_length

        # Utilisation automatique du GPU si disponible, sinon CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Chargement du tokenizer (convertit le texte en tokens numériques)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Chargement du modèle BERT lui-même (pour générer les embeddings)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Mode évaluation (désactive le calcul des gradients pour aller plus vite)
        self.model.eval()

    @torch.no_grad()  # empêche la création d'un graphe de gradient (plus léger)
    def transform(self, X):
        """
        Fonction principale : transforme une liste de textes (X)
        en matrice de vecteurs (embeddings BERT).
        Chaque texte devient un vecteur de taille 768.
        """
        embs = []  # liste pour stocker les embeddings de chaque batch

        # Nombre total de lots à traiter
        n_batches = int(np.ceil(len(X) / self.batch_size))

        # Boucle sur les lots de textes
        for i in tqdm(range(0, len(X), self.batch_size), total=n_batches, desc="Encodage BERT"):
            # Extraction du lot courant
            batch = list(X[i:i+self.batch_size])

            # Tokenisation du texte :
            # - padding : complète les phrases pour avoir la même longueur
            # - truncation : coupe les textes trop longs
            # - max_length : longueur max (ici 96 tokens)
            # - return_tensors="pt" : retourne des tenseurs PyTorch
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Passage dans le modèle BERT
            outputs = self.model(**inputs)

            # Moyenne des embeddings sur les tokens (pooling moyen)
            # → on obtient un vecteur unique par phrase
            mean_emb = outputs.last_hidden_state.mean(dim=1)

            # On récupère le résultat sur le CPU et on le stocke
            embs.append(mean_emb.cpu().numpy())

        # On regroupe tous les lots en une seule matrice numpy
        return np.vstack(embs)

    def fit(self, X, y=None):
        """
        Fonction vide ici car BERT ne s'entraîne pas dans ce contexte.
        Nécessaire uniquement pour compatibilité avec scikit-learn.
        """
        return self
