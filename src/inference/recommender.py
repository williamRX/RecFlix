import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProfileRecommender:
    """
    Moteur de recommandation basé sur le profil utilisateur.
    Utilise l'espace latent (SVD) pour créer un "centre de gravité" des goûts de l'utilisateur
    et calcule la similarité cosinus avec les films non vus.
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Par défaut, on pointe vers le dossier processed du projet
            self.data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
        else:
            self.data_dir = Path(data_dir)
            
        self.latent_features = None
        self.movies_df = None
        self.movie_id_to_idx = {}
        
        self._load_data()
        
    def _load_data(self):
        """Charge la matrice latente et les métadonnées une seule fois en mémoire."""
        logging.info("Chargement des données en mémoire...")
        latent_path = self.data_dir / "latent_features.npy"
        movies_path = self.data_dir / "checkpoint_2_movies_clean.parquet"
        
        if not latent_path.exists() or not movies_path.exists():
            raise FileNotFoundError("Fichiers de données introuvables. Lancez train_svd_model.py.")
            
        self.latent_features = np.load(latent_path)
        self.movies_df = pd.read_parquet(movies_path)
        
        # Création du dictionnaire de correspondance rapide : movieId -> index (ligne de la matrice)
        self.movie_id_to_idx = {int(row['movieId']): idx for idx, row in self.movies_df.iterrows()}
        logging.info("Données chargées et prêtes pour l'inférence.")

    def get_recommendations(self, user_ratings: List[Dict[str, float]], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Génère des recommandations basées sur l'historique de notes de l'utilisateur.
        
        Args:
            user_ratings: Liste de dictionnaires ex: [{"movieId": 1, "rating": 5.0}, ...]
            top_n: Nombre de recommandations à retourner.
            
        Returns:
            Liste des top N films recommandés.
        """
        if not user_ratings:
            logging.warning("Aucune note fournie. Retour de recommandations populaires (non implémenté ici).")
            return []

        # 1. Extraction des vecteurs et calcul des poids
        rated_vectors = []
        weights = []
        rated_movie_ids = set()
        
        for item in user_ratings:
            m_id = int(item['movieId'])
            rating = float(item['rating'])
            
            if m_id in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[m_id]
                rated_vectors.append(self.latent_features[idx])
                
                # Centrage de la note autour de 0 (les notes vont de 0.5 à 5.0)
                # Un rating de 5 donne un poids de +2.0 (tire le profil vers ce film)
                # Un rating de 1 donne un poids de -2.0 (repousse le profil loin de ce film)
                weight = rating - 3.0
                weights.append(weight)
                rated_movie_ids.add(m_id)
            else:
                logging.warning(f"Film ID {m_id} inconnu dans la base. Ignoré.")
                
        if not rated_vectors:
            logging.warning("Aucun des films notés n'a été trouvé dans la base.")
            return []
            
        # 2. Création du Profil Utilisateur (Vecteur 150D)
        rated_vectors = np.array(rated_vectors)
        weights = np.array(weights)
        
        # On multiplie chaque vecteur par son poids, puis on fait la somme.
        # La division par la somme absolue des poids évite des valeurs extrêmes.
        # Mais pour la similarité cosinus (qui ne regarde que l'angle), la somme simple suffit.
        user_profile_vector = np.sum(rated_vectors * weights[:, np.newaxis], axis=0)
        
        # 3. Filtrage des films déjà vus
        all_indices = np.arange(self.latent_features.shape[0])
        rated_indices = [self.movie_id_to_idx[m_id] for m_id in rated_movie_ids]
        
        # Masque booléen : True si le film N'A PAS été vu
        unseen_mask = np.ones(self.latent_features.shape[0], dtype=bool)
        unseen_mask[rated_indices] = False
        
        unseen_indices = all_indices[unseen_mask]
        unseen_features = self.latent_features[unseen_indices]
        
        # 4. Calcul de la similarité cosinus
        # cosine_similarity attend des matrices 2D, on reshape notre vecteur 1D
        similarities = cosine_similarity(user_profile_vector.reshape(1, -1), unseen_features)[0]
        
        # 5. Récupération des Top N
        # argsort trie par ordre croissant, on prend les N derniers et on inverse l'ordre
        top_n_relative_indices = similarities.argsort()[-top_n:][::-1]
        
        recommendations = []
        for rel_idx in top_n_relative_indices:
            real_idx = unseen_indices[rel_idx]
            score = similarities[rel_idx]
            movie_row = self.movies_df.iloc[real_idx]
            
            recommendations.append({
                "movieId": int(movie_row['movieId']),
                "title": str(movie_row['title']),
                "similarity_score": float(round(score, 4)),
                "director": str(movie_row['director'])
            })
            
        return recommendations

if __name__ == "__main__":
    # Test Rapide de l'Inférence
    print("=== Démarrage du Test du Recommender ===")
    recommender = ProfileRecommender()
    
    # Simulation d'un utilisateur qui a adoré Toy Story (Animation) et détesté un film d'horreur
    # 1: Toy Story (1995)
    # 356: Forrest Gump (1994)
    # 296: Pulp Fiction (1994)
    test_ratings = [
        {"movieId": 1, "rating": 5.0},   # Adore Toy Story
        {"movieId": 3114, "rating": 5.0}, # Adore Toy Story 2 (1999)
        {"movieId": 296, "rating": 1.0}   # Déteste Pulp Fiction (Crime, Thriller)
    ]
    
    print(f"Profil Utilisateur Test : {test_ratings}")
    recs = recommender.get_recommendations(test_ratings, top_n=5)
    
    print("\n=== Top 5 Recommandations ===")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['title']} (Score: {r['similarity_score']}) - Réalisateur: {r['director']}")
