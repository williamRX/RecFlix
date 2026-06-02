import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MovieLensDataset(Dataset):
    """
    Dataset PyTorch pour l'architecture Two-Tower sur MovieLens 1M.
    
    Fournit les indices catégoriels pour les couches d'embeddings (sexe, âge, métier, code postal, décennie)
    et les tenseurs continus de caractéristiques comportementales et statistiques.
    """
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        user_demographics: pd.DataFrame,  # DataFrame indexé par user_idx contenant les colonnes d'indices catégoriels
        user_behavioral_features: np.ndarray,  # Matrice (num_users, 48)
        movie_categories: pd.DataFrame,      # DataFrame indexé par movie_idx contenant les colonnes d'indices catégoriels
        movie_behavioral_features: np.ndarray,  # Matrice (num_movies, 20)
        user_to_idx: Dict[int, int],
        movie_to_idx: Dict[int, int]
    ) -> None:
        """
        Initialise le Dataset.
        """
        self.ratings_df = ratings_df.reset_index(drop=True)
        self.user_demographics = user_demographics
        self.user_behavioral_features = user_behavioral_features
        self.movie_categories = movie_categories
        self.movie_behavioral_features = movie_behavioral_features
        self.user_to_idx = user_to_idx
        self.movie_to_idx = movie_to_idx

    def __len__(self) -> int:
        return len(self.ratings_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.ratings_df.iloc[idx]
        user_id = int(row['userId'])
        movie_id = int(row['movieId'])
        rating = float(row['rating'])

        # Récupération des indices séquentiels principaux
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]

        # Données démographiques catégorielles de l'utilisateur
        u_demo = self.user_demographics.iloc[user_idx]
        
        # Données catégorielles du film (décennie)
        m_cat = self.movie_categories.iloc[movie_idx]

        return {
            # Tour Utilisateur
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "gender_idx": torch.tensor(int(u_demo['gender_idx']), dtype=torch.long),
            "age_idx": torch.tensor(int(u_demo['age_idx']), dtype=torch.long),
            "occupation_idx": torch.tensor(int(u_demo['occupation_idx']), dtype=torch.long),
            "zip_idx": torch.tensor(int(u_demo['zip_idx']), dtype=torch.long),
            "user_features": torch.tensor(self.user_behavioral_features[user_idx], dtype=torch.float32),

            # Tour Film
            "movie_idx": torch.tensor(movie_idx, dtype=torch.long),
            "decade_idx": torch.tensor(int(m_cat['decade_idx']), dtype=torch.long),
            "movie_features": torch.tensor(self.movie_behavioral_features[movie_idx], dtype=torch.float32),

            # Cible (Note réelle)
            "rating": torch.tensor(rating, dtype=torch.float32)
        }


def extract_decade_from_title(title: str) -> str:
    """
    Extrait l'année depuis les parenthèses du titre et renvoie la décennie (ex: '1990s').
    """
    match = re.search(r'\((\d{4})\)', title)
    if match:
        year = int(match.group(1))
        decade = (year // 10) * 10
        return f"{decade}s"
    return "Unknown"


def clean_zip(z: Any) -> int:
    """
    Garder le premier chiffre du code postal (0 à 9), repli sur 0 en cas de caractère non numérique.
    """
    z_str = str(z).strip()
    if z_str and z_str[0].isdigit():
        return int(z_str[0])
    return 0


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 256,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Charge et prépare l'ensemble des données MovieLens 1M.
    Calcule les embeddings démographiques et les métriques comportementales avancées.
    
    Returns:
        Un tuple (train_loader, val_loader, metadata_dict)
    """
    raw_dir = data_dir / "raw" / "ml-1m"
    
    # Vérification des fichiers .dat de MovieLens 1M
    for filename in ["users.dat", "movies.dat", "ratings.dat"]:
        if not (raw_dir / filename).exists():
            raise FileNotFoundError(f"Fichier requis introuvable dans {raw_dir}: {filename}")

    # 1. Chargement des fichiers avec délimiteur '::' (moteur python obligatoire pour separator de >1 char)
    logging.info("Chargement de users.dat...")
    users_df = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zip_code"],
        encoding="ISO-8859-1"
    )

    logging.info("Chargement de movies.dat...")
    movies_df = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="ISO-8859-1"
    )

    logging.info("Chargement de ratings.dat...")
    ratings_df = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="ISO-8859-1"
    )

    # 2. Création des mappings séquentiels des IDs principaux
    unique_users = sorted(users_df['userId'].unique())
    unique_movies = sorted(movies_df['movieId'].unique())

    user_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
    idx_to_user = {idx: int(uid) for idx, uid in enumerate(unique_users)}
    movie_to_idx = {int(mid): idx for idx, mid in enumerate(unique_movies)}
    idx_to_movie = {idx: int(mid) for idx, mid in enumerate(unique_movies)}

    # Sauvegarde des index de correspondance pour l'inférence
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    mappings = {
        "user_to_idx": {str(k): v for k, v in user_to_idx.items()},
        "idx_to_user": {str(k): v for k, v in idx_to_user.items()},
        "movie_to_idx": {str(k): v for k, v in movie_to_idx.items()},
        "idx_to_movie": {str(k): v for k, v in idx_to_movie.items()}
    }
    with open(models_dir / "twotower_mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    logging.info(f"Mappings d'index sauvegardés dans {models_dir / 'twotower_mappings.json'}")

    # 3. Extraction dynamique des genres et des décennies
    # Genres
    all_genres = set()
    for genres_str in movies_df['genres'].dropna():
        for g in str(genres_str).split('|'):
            if g.strip():
                all_genres.add(g.strip())
    genres_list = sorted(list(all_genres))
    genre_to_col = {genre: i for i, genre in enumerate(genres_list)}

    # Décennies
    movies_df['decade'] = movies_df['title'].apply(extract_decade_from_title)
    decades_list = sorted(list(movies_df['decade'].unique()))
    decade_to_idx = {decade: i for i, decade in enumerate(decades_list)}

    # 4. Encodage des caractéristiques démographiques (catégoriel)
    # Sexe : F -> 0, M -> 1
    users_df['gender_idx'] = users_df['gender'].map({'F': 0, 'M': 1}).fillna(0).astype(int)
    
    # Âge : Mapper les valeurs uniques [1, 18, 25, 35, 45, 50, 56] vers 0..6
    unique_ages = sorted(users_df['age'].unique())
    age_to_idx = {age: i for i, age in enumerate(unique_ages)}
    users_df['age_idx'] = users_df['age'].map(age_to_idx).fillna(0).astype(int)
    
    # Métier : Déjà séquentiel (0 à 20)
    users_df['occupation_idx'] = users_df['occupation'].fillna(0).astype(int)
    
    users_df['zip_idx'] = users_df['zip_code'].apply(clean_zip).astype(int)

    # DataFrame final des catégories utilisateurs indexé séquentiellement
    users_df['user_idx'] = users_df['userId'].map(user_to_idx)
    user_demographics = users_df.sort_values('user_idx')[['gender_idx', 'age_idx', 'occupation_idx', 'zip_idx']]

    # Catégories de films indexées séquentiellement
    movies_df['movie_idx'] = movies_df['movieId'].map(movie_to_idx)
    movies_df['decade_idx'] = movies_df['decade'].map(decade_to_idx)
    movie_categories = movies_df.sort_values('movie_idx')[['decade_idx']]

    # 5. Split d'entraînement/validation
    logging.info("Division du dataset en Train et Validation...")
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(ratings_df))
    val_size = int(len(ratings_df) * val_split)
    
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]
    
    ratings_train = ratings_df.iloc[train_indices].copy()
    ratings_val = ratings_df.iloc[val_indices].copy()
    
    ratings_train = ratings_train[ratings_train['userId'].isin(user_to_idx) & ratings_train['movieId'].isin(movie_to_idx)]
    ratings_val = ratings_val[ratings_val['userId'].isin(user_to_idx) & ratings_val['movieId'].isin(movie_to_idx)]

    # 6. Feature Engineering Comportemental
    logging.info("Calcul des caractéristiques comportementales...")
    
    # Jointure temporaire Train avec genres et décennies de films
    ratings_joined = ratings_train.merge(movies_df[['movieId', 'genres', 'decade']], on='movieId', how='left')

    # A. Statistiques globales utilisateurs (Train uniquement)
    user_stats = ratings_train.groupby('userId').agg(
        user_mean_rating=('rating', 'mean'),
        user_rating_count=('rating', 'count')
    ).reset_index()
    
    global_mean_rating = ratings_train['rating'].mean()
    user_stats['user_mean_rating'] = user_stats['user_mean_rating'].fillna(global_mean_rating)
    user_stats['user_rating_count'] = user_stats['user_rating_count'].fillna(0.0)

    # Normalisation Z-Score des stats globales utilisateur
    mean_val_u = user_stats['user_mean_rating'].mean()
    std_val_u = user_stats['user_mean_rating'].std() + 1e-8
    count_mean_u = user_stats['user_rating_count'].mean()
    count_std_u = user_stats['user_rating_count'].std() + 1e-8
    
    user_stats['user_mean_rating_norm'] = (user_stats['user_mean_rating'] - mean_val_u) / std_val_u
    user_stats['user_rating_count_norm'] = (user_stats['user_rating_count'] - count_mean_u) / count_std_u

    user_stats_dict = {
        int(row['userId']): (row['user_mean_rating_norm'], row['user_rating_count_norm'], row['user_mean_rating'])
        for _, row in user_stats.iterrows()
    }

    # B. Profil d'intérêt par genre (Proportions)
    # Explode des genres pour grouper
    ratings_exploded = ratings_joined.assign(genre=ratings_joined['genres'].str.split('|')).explode('genre')
    ratings_exploded = ratings_exploded[ratings_exploded['genre'].isin(genre_to_col)] # Filtrer genres inconnus
    
    user_genre_counts = ratings_exploded.groupby(['userId', 'genre']).size().unstack(fill_value=0)
    # Division par le nombre total de films notés par utilisateur pour obtenir des proportions
    user_total_counts = ratings_train.groupby('userId').size()
    user_genre_proportions = user_genre_counts.div(user_total_counts, axis=0).fillna(0.0)
    user_genre_proportions = user_genre_proportions.reindex(columns=genres_list, fill_value=0.0)

    # C. Profil d'appréciation relative par genre (Satisfaction relative)
    user_genre_means = ratings_exploded.groupby(['userId', 'genre'])['rating'].mean().unstack(fill_value=np.nan)
    # Écart : moyenne genre - moyenne globale de l'utilisateur
    user_global_means = user_stats.set_index('userId')['user_mean_rating']
    user_genre_satisfactions = user_genre_means.sub(user_global_means, axis=0).fillna(0.0)
    user_genre_satisfactions = user_genre_satisfactions.reindex(columns=genres_list, fill_value=0.0)

    # D. Profil d'affinité temporelle (Proportions de décennies)
    user_decade_counts = ratings_joined.groupby(['userId', 'decade']).size().unstack(fill_value=0)
    user_decade_proportions = user_decade_counts.div(user_total_counts, axis=0).fillna(0.0)
    user_decade_proportions = user_decade_proportions.reindex(columns=decades_list, fill_value=0.0)

    # E. Construction de la matrice utilisateur finale (num_users, 48)
    # Colonnes : [user_count_norm (1), user_mean_norm (1), genre_proportions (18), genre_satisfactions (18), decade_proportions (10)]
    num_users = len(user_to_idx)
    user_behavioral_features = np.zeros((num_users, 48), dtype=np.float32)

    for uid, u_idx in user_to_idx.items():
        # Stats globales
        norm_mean, norm_count, raw_mean = user_stats_dict.get(uid, (0.0, 0.0, global_mean_rating))
        user_behavioral_features[u_idx, 0] = norm_count
        user_behavioral_features[u_idx, 1] = norm_mean
        
        # Proportions de genres
        if uid in user_genre_proportions.index:
            user_behavioral_features[u_idx, 2:20] = user_genre_proportions.loc[uid].values
            
        # Satisfaction relative
        if uid in user_genre_satisfactions.index:
            user_behavioral_features[u_idx, 20:38] = user_genre_satisfactions.loc[uid].values
            
        # Affinité temporelle (décennies)
        if uid in user_decade_proportions.index:
            user_behavioral_features[u_idx, 38:48] = user_decade_proportions.loc[uid].values

    # F. Statistiques globales films (Train uniquement)
    movie_stats = ratings_train.groupby('movieId').agg(
        movie_mean_rating=('rating', 'mean'),
        movie_rating_count=('rating', 'count')
    ).reset_index()

    movie_stats['movie_mean_rating'] = movie_stats['movie_mean_rating'].fillna(global_mean_rating)
    movie_stats['movie_rating_count'] = movie_stats['movie_rating_count'].fillna(0.0)

    mean_val_m = movie_stats['movie_mean_rating'].mean()
    std_val_m = movie_stats['movie_mean_rating'].std() + 1e-8
    count_mean_m = movie_stats['movie_rating_count'].mean()
    count_std_m = movie_stats['movie_rating_count'].std() + 1e-8

    movie_stats['movie_mean_rating_norm'] = (movie_stats['movie_mean_rating'] - mean_val_m) / std_val_m
    movie_stats['movie_rating_count_norm'] = (movie_stats['movie_rating_count'] - count_mean_m) / count_std_m

    movie_stats_dict = {
        int(row['movieId']): (row['movie_mean_rating_norm'], row['movie_rating_count_norm'])
        for _, row in movie_stats.iterrows()
    }

    # G. Construction de la matrice film finale (num_movies, 20)
    # Colonnes : [multi_hot_genres (18), movie_mean_norm (1), movie_count_norm (1)]
    num_movies = len(movie_to_idx)
    movie_behavioral_features = np.zeros((num_movies, 20), dtype=np.float32)

    for _, row in movies_df.iterrows():
        mid = int(row['movieId'])
        if mid in movie_to_idx:
            m_idx = movie_to_idx[mid]
            
            # Multi-hot des genres
            genres_str = row.get('genres', '')
            if pd.notna(genres_str):
                for g in str(genres_str).split('|'):
                    g_clean = g.strip()
                    if g_clean in genre_to_col:
                        movie_behavioral_features[m_idx, genre_to_col[g_clean]] = 1.0
            
            # Stats de notation normées
            m_mean, m_count = movie_stats_dict.get(mid, (0.0, 0.0))
            movie_behavioral_features[m_idx, -2] = m_mean
            movie_behavioral_features[m_idx, -1] = m_count

    # 7. Sauvegarde des métadonnées pour l'inférence
    metadata = {
        "genres": genres_list,
        "decades": decades_list,
        "user_stats_normalization": {
            "mean_rating": float(mean_val_u),
            "std_rating": float(std_val_u),
            "mean_count": float(count_mean_u),
            "std_count": float(count_std_u)
        },
        "movie_stats_normalization": {
            "mean_rating": float(mean_val_m),
            "std_rating": float(std_val_m),
            "mean_count": float(count_mean_m),
            "std_count": float(count_std_m)
        },
        "global_mean_rating": float(global_mean_rating),
        "age_categories": [int(a) for a in unique_ages]
    }
    with open(models_dir / "twotower_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    logging.info(f"Métadonnées de normalisation enregistrées dans {models_dir / 'twotower_metadata.json'}")

    # 8. Instanciation des Datasets et DataLoaders PyTorch
    train_dataset = MovieLensDataset(
        ratings_df=ratings_train,
        user_demographics=user_demographics,
        user_behavioral_features=user_behavioral_features,
        movie_categories=movie_categories,
        movie_behavioral_features=movie_behavioral_features,
        user_to_idx=user_to_idx,
        movie_to_idx=movie_to_idx
    )
    
    val_dataset = MovieLensDataset(
        ratings_df=ratings_val,
        user_demographics=user_demographics,
        user_behavioral_features=user_behavioral_features,
        movie_categories=movie_categories,
        movie_behavioral_features=movie_behavioral_features,
        user_to_idx=user_to_idx,
        movie_to_idx=movie_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    metadata_dict = {
        "num_users": len(unique_users),
        "num_movies": len(unique_movies),
        "num_decades": len(decades_list),
        "user_features_dim": user_behavioral_features.shape[1],
        "movie_features_dim": movie_behavioral_features.shape[1],
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx
    }

    logging.info(f"DataLoaders initialisés avec succès. Train: {len(train_dataset)} notes, Val: {len(val_dataset)} notes.")
    return train_loader, val_loader, metadata_dict


if __name__ == "__main__":
    # Test rapide de lecture du dataset MovieLens 1M
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "data"
    
    try:
        t_loader, v_loader, meta = get_dataloaders(DATA_PATH, batch_size=32)
        print("\n--- Test du Loader V2 (MovieLens 1M) ---")
        print(f"Nombre d'utilisateurs uniques  : {meta['num_users']}")
        print(f"Nombre de films uniques        : {meta['num_movies']}")
        print(f"Nombre de décennies uniques     : {meta['num_decades']}")
        print(f"Dimension vecteur comportemental user : {meta['user_features_dim']} (attendu : 48)")
        print(f"Dimension vecteur comportemental movie: {meta['movie_features_dim']} (attendu : 20)")
        
        # Récupération d'un batch
        batch = next(iter(t_loader))
        print("Taille des tenseurs du premier batch :")
        for k, v in batch.items():
            print(f" - {k.ljust(15)}: {v.shape} | type: {v.dtype}")
        print("✅ Le loader est prêt et validé pour MovieLens 1M !")
    except Exception as error:
        print(f"❌ Erreur lors du chargement : {error}")
