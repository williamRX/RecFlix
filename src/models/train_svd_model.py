import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
from pathlib import Path
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def train_svd_model(n_components=150):
    """
    Entraîne un modèle SVD pour réduire la dimensionnalité de la matrice de features,
    puis utilise t-SNE pour créer une projection 2D des films.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info("Étape 1 : Chargement des données...")
    matrix_path = PROC_DIR / "features_matrix.npz"
    metadata_path = PROC_DIR / "checkpoint_2_movies_clean.parquet"
    
    if not matrix_path.exists() or not metadata_path.exists():
        logging.error("Fichiers introuvables. Assurez-vous d'avoir lancé build_features.py.")
        return
        
    feature_matrix = sp.load_npz(matrix_path)
    movies_df = pd.read_parquet(metadata_path)
    
    logging.info(f"Matrice chargée avec succès. Dimensions : {feature_matrix.shape}")
    
    # ==========================================
    # 1. Réduction de dimension avec TruncatedSVD
    # ==========================================
    logging.info(f"Étape 2 : Réduction SVD ({n_components} composantes)...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_features = svd.fit_transform(feature_matrix)
    
    variance = svd.explained_variance_ratio_.sum()
    logging.info(f"Variance expliquée cumulée : {variance:.2%} (Plus c'est élevé, mieux c'est)")
    
    logging.info("Sauvegarde du modèle SVD et des features latentes...")
    joblib.dump(svd, MODELS_DIR / "svd_model.joblib")
    np.save(PROC_DIR / "latent_features.npy", latent_features)
    
    # ==========================================
    # 2. Cartographie 2D avec t-SNE
    # ==========================================
    logging.info("Étape 3 : Cartographie 3D avec t-SNE (Cela peut prendre quelques minutes)...")
    # On applique t-SNE sur les features latentes (150D) plutôt que sur la matrice originale
    # pour gagner énormément de temps et de précision.
    tsne = TSNE(n_components=3, init='pca', learning_rate='auto', random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(latent_features)
    
    logging.info("Création et sauvegarde de la carte 3D...")
    movies_3d = pd.DataFrame({
        'movieId': movies_df['movieId'],
        'title': movies_df['title'],
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'z': tsne_results[:, 2]
    })
    
    movies_3d.to_parquet(PROC_DIR / "movies_3d_map.parquet", index=False)
    
    logging.info("🎉 Modèles SVD et Cartographie 3D générés avec succès !")

if __name__ == "__main__":
    train_svd_model(n_components=150)
