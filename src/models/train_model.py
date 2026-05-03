import pandas as pd
import scipy.sparse as sp
import logging
from pathlib import Path
import joblib
import json
from sklearn.neighbors import NearestNeighbors

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def train_model():
    """Entraîne et sauvegarde le modèle de similarité (NearestNeighbors)."""
    
    # Création du dossier models s'il n'existe pas
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info("Étape 1 : Chargement de la matrice de features...")
    matrix_path = PROC_DIR / "features_matrix.npz"
    if not matrix_path.exists():
        logging.error(f"Fichier introuvable : {matrix_path}. Veuillez lancer build_features.py d'abord.")
        return
    
    feature_matrix = sp.load_npz(matrix_path)
    logging.info(f"Matrice chargée avec succès. Dimensions : {feature_matrix.shape}")
    
    logging.info("Étape 2 : Entraînement du modèle NearestNeighbors...")
    # On utilise la distance cosinus pour mesurer la similarité
    # L'algorithme brute est le plus adapté pour des matrices creuses avec la distance cosinus
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(feature_matrix)
    
    logging.info("Étape 3 : Sauvegarde du modèle...")
    model_path = MODELS_DIR / "knn_model.joblib"
    joblib.dump(model_knn, model_path)
    logging.info(f"Modèle sauvegardé dans : {model_path}")
    
    logging.info("Étape 4 : Création du dictionnaire de correspondance des index...")
    # Pour que l'API puisse retrouver un film, il faut lier l'index de la matrice (0 à N) 
    # au vrai movieId du dataset.
    movies_clean_path = PROC_DIR / "checkpoint_2_movies_clean.parquet"
    if not movies_clean_path.exists():
        logging.error(f"Fichier introuvable : {movies_clean_path}. Veuillez lancer build_features.py d'abord.")
        return
        
    df_clean = pd.read_parquet(movies_clean_path)
    
    # On crée deux dictionnaires de correspondance
    movie_id_to_idx = {int(row['movieId']): idx for idx, row in df_clean.iterrows()}
    idx_to_movie_id = {idx: int(row['movieId']) for idx, row in df_clean.iterrows()}
    
    mapping = {
        "movie_id_to_idx": movie_id_to_idx,
        "idx_to_movie_id": idx_to_movie_id
    }
    
    mapping_path = MODELS_DIR / "movie_indices_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f)
        
    logging.info(f"Mapping sauvegardé dans : {mapping_path}")
    logging.info("🎉 Le moteur de recommandation est prêt !")

if __name__ == "__main__":
    train_model()
