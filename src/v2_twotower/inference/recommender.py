import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from src.v2_twotower.data_loader import get_dataloaders, clean_zip, extract_decade_from_title
from src.v2_twotower.model import TwoTowerModel

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TwoTowerRecommender:
    """
    Moteur de recommandation en ligne pour le modèle Two-Tower V2 (MovieLens 1M).
    
    Il pré-calcule et met en cache les embeddings de tous les films au démarrage,
    puis utilise des opérations matricielles rapides (produit scalaire) pour générer
    des recommandations personnalisées en quelques millisecondes.
    """
    
    def __init__(self, data_dir: Path, models_dir: Path, device: torch.device) -> None:
        """
        Initialise le moteur de recommandation.
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.device = device
        
        # 1. Chargement des métadonnées et mappings d'entraînement
        metadata_path = models_dir / "twotower_metadata.json"
        mappings_path = models_dir / "twotower_mappings.json"
        model_weights_path = models_dir / "twotower_best_model.pth"
        
        if not (metadata_path.exists() and mappings_path.exists() and model_weights_path.exists()):
            raise FileNotFoundError("Fichiers de modèle ou de métadonnées manquants dans /models/. Veuillez lancer train.py d'abord.")
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        with open(mappings_path, 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
            
        # Reconstruction des dictionnaires de mapping (avec types corrects)
        self.user_to_idx = {int(k): v for k, v in self.mappings["user_to_idx"].items()}
        self.idx_to_user = {int(k): int(v) for k, v in self.mappings["idx_to_user"].items()}
        self.movie_to_idx = {int(k): v for k, v in self.mappings["movie_to_idx"].items()}
        self.idx_to_movie = {int(k): int(v) for k, v in self.mappings["idx_to_movie"].items()}
        
        self.genres_list = self.metadata["genres"]
        self.decades_list = self.metadata["decades"]
        self.age_categories = self.metadata["age_categories"]
        
        # Mapping inverse de l'âge
        self.age_to_idx = {age: i for i, age in enumerate(self.age_categories)}
        self.decade_to_idx = {decade: i for i, decade in enumerate(self.decades_list)}
        
        # 2. Instanciation et chargement du modèle
        logging.info("Chargement du modèle Two-Tower et des poids...")
        self.model = TwoTowerModel(
            num_users=len(self.user_to_idx),
            num_items=len(self.movie_to_idx),
            num_decades=len(self.decades_list),
            user_metadata_dim=self.metadata.get("user_features_dim", 48),  # 48 dimensions comportementales
            item_metadata_dim=self.metadata.get("item_features_dim", 20)   # 20 dimensions film
        )
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 3. Récupération des DataLoaders ou chargement du cache d'inférence
        cache_path = models_dir / "twotower_inference_cache.pth"
        if cache_path.exists():
            logging.info("Chargement du cache d'inférence Two-Tower pré-calculé...")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.user_seen_movies = cache["user_seen_movies"]
            self.popular_movie_ids = cache["popular_movie_ids"]
            self.user_history_ratings = cache["user_history_ratings"]
            self._movie_decade_idx = cache["movie_decade_idx"]
            self._movie_behavioral_features = cache["movie_behavioral_features"]
            self._user_demographics = cache["user_demographics"]
            self._user_behavioral_features = cache["user_behavioral_features"]
        else:
            logging.info("Cache d'inférence introuvable. Calcul des caractéristiques depuis le dataset brut...")
            _, val_loader, _ = get_dataloaders(self.data_dir, batch_size=256, val_split=0.2)
            val_dataset = val_loader.dataset
            
            # Cache des notes d'entraînement pour pouvoir filtrer les films déjà vus
            ratings_train = val_dataset.ratings_df
            self.user_seen_movies = ratings_train.groupby('userId')['movieId'].apply(set).to_dict()
            self.popular_movie_ids = ratings_train['movieId'].value_counts().head(24).index.tolist()
            
            # Reconstruction de l'historique des évaluations par utilisateur
            user_history_ratings = {}
            for _, row in ratings_train.iterrows():
                u_id = int(row['userId'])
                m_id = int(row['movieId'])
                r = float(row['rating'])
                if u_id not in user_history_ratings:
                    user_history_ratings[u_id] = []
                user_history_ratings[u_id].append({"movieId": m_id, "rating": r})
            self.user_history_ratings = user_history_ratings
            
            self._movie_decade_idx = val_dataset.movie_categories['decade_idx'].values
            self._movie_behavioral_features = val_dataset.movie_behavioral_features
            self._user_demographics = val_dataset.user_demographics
            self._user_behavioral_features = val_dataset.user_behavioral_features
            
            # Sauvegarde automatique du cache pour les prochains démarrages ou déploiements
            try:
                logging.info(f"Création et sauvegarde du cache d'inférence dans {cache_path}...")
                torch.save({
                    "user_seen_movies": self.user_seen_movies,
                    "popular_movie_ids": self.popular_movie_ids,
                    "user_history_ratings": self.user_history_ratings,
                    "movie_decade_idx": self._movie_decade_idx,
                    "movie_behavioral_features": self._movie_behavioral_features,
                    "user_demographics": self._user_demographics,
                    "user_behavioral_features": self._user_behavioral_features
                }, cache_path)
            except Exception as e:
                logging.error(f"Impossible de sauvegarder le cache d'inférence : {e}")

        # 4. Pré-calcul des embeddings de tous les films (Item Tower)
        self._precompute_movie_embeddings()

    def _precompute_movie_embeddings(self) -> None:
        """
        Passe tous les films dans la Item Tower et met en cache leurs représentations normalisées L2.
        """
        logging.info("Pré-calcul des représentations (embeddings) de tous les films...")
        
        num_movies = len(self.movie_to_idx)
        movie_idx = torch.arange(num_movies, dtype=torch.long).to(self.device)
        
        # Extraction des catégories (décennie) et features numériques précalculées pour chaque film
        decade_idx = torch.tensor(self._movie_decade_idx, dtype=torch.long).to(self.device)
        movie_features = torch.tensor(self._movie_behavioral_features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            raw_movie_embs = self.model.item_tower(movie_idx, decade_idx, movie_features)
            # Normalisation L2 pour le calcul de similarité cosinus direct (produit matriciel)
            self.movie_embeddings = F.normalize(raw_movie_embs, p=2, dim=1)
            
        logging.info(f"Embeddings de {num_movies} films mis en cache dans le device {self.device}.")

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Génère des recommandations pour un utilisateur existant dans la base de données (Warm Start).
        
        Args:
            user_id: ID de l'utilisateur réel.
            top_n: Nombre de recommandations à renvoyer.
            
        Returns:
            Liste de dictionnaires contenant les films recommandés et leurs scores.
        """
        if user_id not in self.user_to_idx:
            raise KeyError(f"Utilisateur {user_id} inconnu dans la base. Utilisez recommend_cold_start.")
            
        user_idx = self.user_to_idx[user_id]
        
        # 1. Récupération des caractéristiques catégorielles et comportementales
        u_demo = self._user_demographics.iloc[user_idx]
        user_features_np = self._user_behavioral_features[user_idx]
        
        # 2. Conversion en tenseurs
        user_idx_t = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        gender_idx_t = torch.tensor([int(u_demo['gender_idx'])], dtype=torch.long).to(self.device)
        age_idx_t = torch.tensor([int(u_demo['age_idx'])], dtype=torch.long).to(self.device)
        occupation_idx_t = torch.tensor([int(u_demo['occupation_idx'])], dtype=torch.long).to(self.device)
        zip_idx_t = torch.tensor([int(u_demo['zip_idx'])], dtype=torch.long).to(self.device)
        user_features_t = torch.tensor(user_features_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Calcul de la représentation utilisateur
        with torch.no_grad():
            u_emb = self.model.user_tower(
                user_idx_t, gender_idx_t, age_idx_t, occupation_idx_t, zip_idx_t, user_features_t
            )
            u_emb_norm = F.normalize(u_emb, p=2, dim=1) # (1, 32)
            
        # 4. Calcul de la similarité cosinus matricielle contre tous les films
        # (1, 32) x (32, num_movies) -> (num_movies,)
        similarities = torch.matmul(u_emb_norm, self.movie_embeddings.T).squeeze(0)
        
        # Projection de la similarité sur l'échelle des notes réelles
        pred_ratings = self.model.min_rating + (self.model.max_rating - self.model.min_rating) * (similarities + 1.0) / 2.0
        scores = pred_ratings.cpu().numpy()
        
        # 5. Filtrage des films déjà vus et tri des scores
        seen_movies = self.user_seen_movies.get(user_id, set())
        
        # Liste des films recommandés candidats
        candidates = []
        for m_idx in range(len(scores)):
            m_id = self.idx_to_movie[m_idx]
            if m_id not in seen_movies:
                candidates.append((m_id, float(scores[m_idx])))
                
        # Tri par score décroissant
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]
        
        # 6. Reconstruction de la réponse avec métadonnées
        # Récupération rapide des titres depuis movies_df
        movies_raw_path = self.data_dir / "raw" / "ml-1m" / "movies.dat"
        movies_meta_df = pd.read_csv(
            movies_raw_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="ISO-8859-1"
        )
        movies_meta_df = movies_meta_df.set_index('movieId')
        
        recommendations = []
        for m_id, score in candidates:
            row = movies_meta_df.loc[m_id]
            recommendations.append({
                "movieId": m_id,
                "title": row['title'],
                "genres": row['genres'],
                "score": round(score, 3)
            })
            
        return recommendations

    def recommend_cold_start(
        self,
        gender: str,        # 'F' ou 'M'
        age: int,           # Valeurs réelles de MovieLens (ex: 25)
        occupation: int,    # Valeur de 0 à 20
        zip_code: str,      # ex: '90210'
        liked_movie_ids: Optional[List[int]] = None,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Génère des recommandations pour un nouvel utilisateur (Cold Start) ou un appel d'API Hugging Face.
        Reconstruit implicitement les caractéristiques démographiques et le profil comportemental.
        """
        # 1. Encodage démographique
        gender_idx = 1 if gender == 'M' else 0
        
        # Trouver la tranche d'âge la plus proche
        age_idx = 0
        if age in self.age_to_idx:
            age_idx = self.age_to_idx[age]
        else:
            # Recherche de la tranche d'âge la plus proche
            distances = [abs(a - age) for a in self.age_categories]
            age_idx = np.argmin(distances)
            
        occupation_idx = occupation if (0 <= occupation <= 20) else 0
        zip_idx = clean_zip(zip_code)
        
        # 2. Construction dynamique du profil comportemental de dimension 48
        user_features_np = np.zeros(48, dtype=np.float32)
        
        # S'il n'y a pas de films aimés, le comportement est neutre (Z-score = 0.0)
        # S'il y a des films aimés, on recrée le profil d'intérêt thématique, de satisfaction et temporel :
        if liked_movie_ids:
            # Charger les infos des films aimés
            movies_raw_path = self.data_dir / "raw" / "ml-1m" / "movies.dat"
            movies_meta_df = pd.read_csv(
                movies_raw_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="ISO-8859-1"
            )
            movies_meta_df['decade'] = movies_meta_df['title'].apply(extract_decade_from_title)
            liked_movies = movies_meta_df[movies_meta_df['movieId'].isin(liked_movie_ids)]
            
            # Nombre de notes simulées
            count = len(liked_movie_ids)
            mean_rating = 5.0 # On suppose qu'il a adoré ces films (note de 5/5)
            
            # Normalisation avec les stats d'entraînement globales
            norm_mean = (mean_rating - self.metadata["user_stats_normalization"]["mean_rating"]) / self.metadata["user_stats_normalization"]["std_rating"]
            norm_count = (count - self.metadata["user_stats_normalization"]["mean_count"]) / self.metadata["user_stats_normalization"]["std_count"]
            user_features_np[0] = norm_count
            user_features_np[1] = norm_mean
            
            # Proportions de genres
            genre_counts = {g: 0 for g in self.genres_list}
            for _, row in liked_movies.iterrows():
                for g in str(row['genres']).split('|'):
                    if g.strip() in genre_counts:
                        genre_counts[g.strip()] += 1
            for i, g in enumerate(self.genres_list):
                user_features_np[2 + i] = genre_counts[g] / count
                
            # Satisfaction relative par genre (les films aimés reçoivent +1.0 par rapport à sa moyenne)
            for i, g in enumerate(self.genres_list):
                if genre_counts[g] > 0:
                    user_features_np[20 + i] = 1.0 # 5.0 - 4.0 = +1.0 d'écart positif
                    
            # Proportions de décennies
            decade_counts = {d: 0 for d in self.decades_list}
            for _, row in liked_movies.iterrows():
                d = row['decade']
                if d in decade_counts:
                    decade_counts[d] += 1
            for i, d in enumerate(self.decades_list):
                user_features_np[38 + i] = decade_counts[d] / count
                
        # 3. Conversion en tenseurs (on utilise l'index utilisateur par défaut 0)
        user_idx_t = torch.tensor([0], dtype=torch.long).to(self.device)
        gender_idx_t = torch.tensor([gender_idx], dtype=torch.long).to(self.device)
        age_idx_t = torch.tensor([age_idx], dtype=torch.long).to(self.device)
        occupation_idx_t = torch.tensor([occupation_idx], dtype=torch.long).to(self.device)
        zip_idx_t = torch.tensor([zip_idx], dtype=torch.long).to(self.device)
        user_features_t = torch.tensor(user_features_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. Passage dans la User Tower
        with torch.no_grad():
            u_emb = self.model.user_tower(
                user_idx_t, gender_idx_t, age_idx_t, occupation_idx_t, zip_idx_t, user_features_t
            )
            u_emb_norm = F.normalize(u_emb, p=2, dim=1)
            
        # 5. Calcul de la similarité matricielle
        similarities = torch.matmul(u_emb_norm, self.movie_embeddings.T).squeeze(0)
        pred_ratings = self.model.min_rating + (self.model.max_rating - self.model.min_rating) * (similarities + 1.0) / 2.0
        scores = pred_ratings.cpu().numpy()
        
        # 6. Filtrage des films déjà aimés en entrée et tri des scores
        liked_set = set(liked_movie_ids) if liked_movie_ids else set()
        candidates = []
        for m_idx in range(len(scores)):
            m_id = self.idx_to_movie[m_idx]
            if m_id not in liked_set:
                candidates.append((m_id, float(scores[m_idx])))
                
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]
        
        # 7. Traduction en titres
        movies_raw_path = self.data_dir / "raw" / "ml-1m" / "movies.dat"
        movies_meta_df = pd.read_csv(
            movies_raw_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="ISO-8859-1"
        )
        movies_meta_df = movies_meta_df.set_index('movieId')
        
        recommendations = []
        for m_id, score in candidates:
            row = movies_meta_df.loc[m_id]
            recommendations.append({
                "movieId": m_id,
                "title": row['title'],
                "genres": row['genres'],
                "score": round(score, 3)
            })
            
        return recommendations

    def recommend_with_feedback(
        self,
        gender: str,
        age: int,
        occupation: int,
        zip_code_or_idx: Any,
        ratings: List[Dict[str, Any]],
        seen_movie_ids: Optional[set] = None,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Génère des recommandations pour un utilisateur (nouveau ou existant) en recalculant 
        dynamiquement son profil comportemental de dimension 48 à partir d'une liste de notes explicites.
        """
        # 1. Encodage démographique
        gender_idx = 1 if gender == 'M' else 0
        
        # Trouver la tranche d'âge la plus proche
        age_idx = 0
        if age in self.age_to_idx:
            age_idx = self.age_to_idx[age]
        else:
            distances = [abs(a - age) for a in self.age_categories]
            age_idx = np.argmin(distances)
            
        occupation_idx = occupation if (0 <= occupation <= 20) else 0
        
        if isinstance(zip_code_or_idx, str):
            zip_idx = clean_zip(zip_code_or_idx)
        else:
            zip_idx = int(zip_code_or_idx)
            
        # 2. Construction dynamique du profil comportemental de dimension 48
        user_features_np = np.zeros(48, dtype=np.float32)
        
        if ratings:
            liked_movie_ids = [r["movieId"] for r in ratings]
            
            # Chargement des métadonnées pour extraire genres et décennies
            movies_raw_path = self.data_dir / "raw" / "ml-1m" / "movies.dat"
            movies_meta_df = pd.read_csv(
                movies_raw_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="ISO-8859-1"
            )
            movies_meta_df['decade'] = movies_meta_df['title'].apply(extract_decade_from_title)
            
            rated_movies_df = movies_meta_df[movies_meta_df['movieId'].isin(liked_movie_ids)].copy()
            
            # Associe les notes utilisateur aux films
            ratings_map = {r["movieId"]: r["rating"] for r in ratings}
            rated_movies_df['user_rating'] = rated_movies_df['movieId'].map(ratings_map)
            
            count = len(ratings)
            mean_rating = np.mean([r["rating"] for r in ratings])
            
            # Normalisation avec les statistiques d'entraînement globales
            norm_mean = (mean_rating - self.metadata["user_stats_normalization"]["mean_rating"]) / self.metadata["user_stats_normalization"]["std_rating"]
            norm_count = (count - self.metadata["user_stats_normalization"]["mean_count"]) / self.metadata["user_stats_normalization"]["std_count"]
            user_features_np[0] = norm_count
            user_features_np[1] = norm_mean
            
            # Compte des genres et calcul de la satisfaction par genre
            genre_counts = {g: 0 for g in self.genres_list}
            genre_ratings_sum = {g: 0.0 for g in self.genres_list}
            for _, row in rated_movies_df.iterrows():
                rating_val = row['user_rating']
                for g in str(row['genres']).split('|'):
                    g_clean = g.strip()
                    if g_clean in genre_counts:
                        genre_counts[g_clean] += 1
                        genre_ratings_sum[g_clean] += rating_val
            
            # Proportions de genres (indices 2 à 19)
            for i, g in enumerate(self.genres_list):
                user_features_np[2 + i] = genre_counts[g] / count
                
            # Satisfaction relative par genre (indices 20 à 37)
            for i, g in enumerate(self.genres_list):
                if genre_counts[g] > 0:
                    genre_avg = genre_ratings_sum[g] / genre_counts[g]
                    user_features_np[20 + i] = genre_avg - mean_rating
                    
            # Proportions de décennies (indices 38 à 47)
            decade_counts = {d: 0 for d in self.decades_list}
            for _, row in rated_movies_df.iterrows():
                d = row['decade']
                if d in decade_counts:
                    decade_counts[d] += 1
            for i, d in enumerate(self.decades_list):
                user_features_np[38 + i] = decade_counts[d] / count
                
        # 3. Passage dans la User Tower
        user_idx_t = torch.tensor([0], dtype=torch.long).to(self.device)
        gender_idx_t = torch.tensor([gender_idx], dtype=torch.long).to(self.device)
        age_idx_t = torch.tensor([age_idx], dtype=torch.long).to(self.device)
        occupation_idx_t = torch.tensor([occupation_idx], dtype=torch.long).to(self.device)
        zip_idx_t = torch.tensor([zip_idx], dtype=torch.long).to(self.device)
        user_features_t = torch.tensor(user_features_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            u_emb = self.model.user_tower(
                user_idx_t, gender_idx_t, age_idx_t, occupation_idx_t, zip_idx_t, user_features_t
            )
            u_emb_norm = F.normalize(u_emb, p=2, dim=1)
            
        # 4. Calcul de la similarité cosinus avec tous les films
        similarities = torch.matmul(u_emb_norm, self.movie_embeddings.T).squeeze(0)
        pred_ratings = self.model.min_rating + (self.model.max_rating - self.model.min_rating) * (similarities + 1.0) / 2.0
        scores = pred_ratings.cpu().numpy()
        
        # 5. Filtrage des films déjà vus ou évalués
        filter_set = seen_movie_ids if seen_movie_ids is not None else set()
        candidates = []
        for m_idx in range(len(scores)):
            m_id = self.idx_to_movie[m_idx]
            if m_id not in filter_set:
                candidates.append((m_id, float(scores[m_idx])))
                
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]
        
        # 6. Traduction en titres de films
        movies_raw_path = self.data_dir / "raw" / "ml-1m" / "movies.dat"
        movies_meta_df = pd.read_csv(
            movies_raw_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="ISO-8859-1"
        )
        movies_meta_df = movies_meta_df.set_index('movieId')
        
        recommendations = []
        for m_id, score in candidates:
            row = movies_meta_df.loc[m_id]
            recommendations.append({
                "movieId": m_id,
                "title": row['title'],
                "genres": row['genres'],
                "score": round(score, 3)
            })
            
        return recommendations


if __name__ == "__main__":
    # Test de l'inférence de bout en bout
    print("=== Démarrage du Test d'Inférence Two-Tower V2 ===")
    
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "data"
    models_path = project_root / "models"
    
    # Sélection de l'accélérateur
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        recommender = TwoTowerRecommender(data_dir=data_path, models_dir=models_path, device=device)
        
        # Test 1 : Utilisateur existant (userId = 1)
        print("\n--- Test 1 : Recommandation pour l'utilisateur ID 1 (Warm Start) ---")
        recs_warm = recommender.recommend_for_user(user_id=1, top_n=5)
        for i, r in enumerate(recs_warm, 1):
            print(f"{i}. {r['title']} | Genres: {r['genres']} | Note prédite: {r['score']:.2f}")
            
        # Test 2 : Nouvel utilisateur, sans historique de films (Cold Start démographique pur)
        # Profil simulé : Femme, 25 ans, College student (occupation 4), Zipcode '90210'
        print("\n--- Test 2 : Nouvel utilisateur (Femme, 25 ans, Étudiante) - Cold Start pur ---")
        recs_cold_pure = recommender.recommend_cold_start(
            gender='F', age=25, occupation=4, zip_code='90210', liked_movie_ids=[], top_n=5
        )
        for i, r in enumerate(recs_cold_pure, 1):
            print(f"{i}. {r['title']} | Genres: {r['genres']} | Note prédite: {r['score']:.2f}")
            
        # Test 3 : Nouvel utilisateur avec historique dynamique
        # Profil similaire qui aime les dessins animés et films d'aventure Disney
        # 1: Toy Story (1995), 588: Aladdin (1992)
        print("\n--- Test 3 : Nouvel utilisateur qui aime Toy Story (1) et Aladdin (588) ---")
        recs_cold_dynamic = recommender.recommend_cold_start(
            gender='F', age=25, occupation=4, zip_code='90210', liked_movie_ids=[1, 588], top_n=5
        )
        for i, r in enumerate(recs_cold_dynamic, 1):
            print(f"{i}. {r['title']} | Genres: {r['genres']} | Note prédite: {r['score']:.2f}")
            
        print("\n✅ Le moteur de recommandation Two-Tower V2 fonctionne parfaitement !")
    except Exception as error:
        logging.error(f"Erreur durant l'inférence : {error}", exc_info=True)
