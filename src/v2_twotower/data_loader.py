import os
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
    Dataset PyTorch personnalisé pour l'architecture de recommandation Two-Tower.
    
    Cette classe prépare et fournit les indices d'utilisateurs et de films,
    ainsi que leurs caractéristiques (features) respectives et la note cible.
    """
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        user_features: np.ndarray,
        movie_features: np.ndarray,
        user_to_idx: Dict[int, int],
        movie_to_idx: Dict[int, int]
    ) -> None:
        """
        Initialise le Dataset.
        
        Args:
            ratings_df: DataFrame contenant les colonnes ['userId', 'movieId', 'rating']
            user_features: Matrice numpy des caractéristiques utilisateurs, indexée par user_idx
            movie_features: Matrice numpy des caractéristiques de films, indexée par movie_idx
            user_to_idx: Dictionnaire de mapping userId -> index séquentiel (0 à U-1)
            movie_to_idx: Dictionnaire de mapping movieId -> index séquentiel (0 à M-1)
        """
        self.ratings_df = ratings_df.reset_index(drop=True)
        self.user_features = user_features
        self.movie_features = movie_features
        self.user_to_idx = user_to_idx
        self.movie_to_idx = movie_to_idx

    def __len__(self) -> int:
        """Retourne le nombre total d'interactions (notes)."""
        return len(self.ratings_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Récupère un échantillon pour l'entraînement ou l'évaluation.
        
        Args:
            idx: Index de la ligne demandée.
            
        Returns:
            Un dictionnaire contenant les tenseurs d'entrée pour les deux tours et la note cible.
        """
        row = self.ratings_df.iloc[idx]
        user_id = int(row['userId'])
        movie_id = int(row['movieId'])
        rating = float(row['rating'])

        # Conversion vers les index séquentiels
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "movie_idx": torch.tensor(movie_idx, dtype=torch.long),
            "user_features": torch.tensor(self.user_features[user_idx], dtype=torch.float32),
            "movie_features": torch.tensor(self.movie_features[movie_idx], dtype=torch.float32),
            "rating": torch.tensor(rating, dtype=torch.float32)
        }


def prepare_features(
    ratings_train: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_to_idx: Dict[int, int],
    movie_to_idx: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Construit les matrices de caractéristiques pour les utilisateurs et les films.
    Calcule des statistiques pour éviter le data leakage (basé uniquement sur le train).
    
    Args:
        ratings_train: DataFrame d'entraînement des interactions.
        movies_df: DataFrame des métadonnées de films.
        user_to_idx: Dictionnaire de mapping userId -> user_idx.
        movie_to_idx: Dictionnaire de mapping movieId -> movie_idx.
        
    Returns:
        Un tuple contenant :
        - user_features (np.ndarray) de dimension (num_users, user_features_dim)
        - movie_features (np.ndarray) de dimension (num_movies, movie_features_dim)
        - user_features_dim (int)
        - movie_features_dim (int)
    """
    num_users = len(user_to_idx)
    num_movies = len(movie_to_idx)
    
    # 1. Caractéristiques Utilisateurs (Moyenne des notes données et nombre total de notes)
    user_stats = ratings_train.groupby('userId').agg(
        user_mean_rating=('rating', 'mean'),
        user_rating_count=('rating', 'count')
    ).reset_index()
    
    # Normalisation des statistiques utilisateurs
    user_mean_global = ratings_train['rating'].mean()
    user_stats['user_mean_rating'] = user_stats['user_mean_rating'].fillna(user_mean_global)
    user_stats['user_rating_count'] = user_stats['user_rating_count'].fillna(0.0)
    
    # Standardisation
    mean_val_u = user_stats['user_mean_rating'].mean()
    std_val_u = user_stats['user_mean_rating'].std() + 1e-8
    count_mean_u = user_stats['user_rating_count'].mean()
    count_std_u = user_stats['user_rating_count'].std() + 1e-8
    
    user_stats['user_mean_rating_norm'] = (user_stats['user_mean_rating'] - mean_val_u) / std_val_u
    user_stats['user_rating_count_norm'] = (user_stats['user_rating_count'] - count_mean_u) / count_std_u
    
    # Construction de la matrice user_features (dim: num_users, 2)
    # Remplissage par défaut (0.0 pour les valeurs normées)
    user_features = np.zeros((num_users, 2), dtype=np.float32)
    for _, row in user_stats.iterrows():
        u_id = int(row['userId'])
        if u_id in user_to_idx:
            u_idx = user_to_idx[u_id]
            user_features[u_idx, 0] = row['user_mean_rating_norm']
            user_features[u_idx, 1] = row['user_rating_count_norm']
            
    # 2. Caractéristiques Films (Genres multi-hot + Moyenne de notes reçues et nombre de votes)
    # Extraction dynamique des genres uniques
    all_genres = set()
    for genres_str in movies_df['genres'].dropna():
        for g in str(genres_str).split('|'):
            if g.strip():
                all_genres.add(g.strip())
    genres_list = sorted(list(all_genres))
    genre_to_col = {genre: i for i, genre in enumerate(genres_list)}
    num_genres = len(genres_list)
    
    # Calcul des stats de notes reçues par film dans l'ensemble train
    movie_stats = ratings_train.groupby('movieId').agg(
        movie_mean_rating=('rating', 'mean'),
        movie_rating_count=('rating', 'count')
    ).reset_index()
    
    # Normalisation des statistiques films
    movie_stats['movie_mean_rating'] = movie_stats['movie_mean_rating'].fillna(user_mean_global)
    movie_stats['movie_rating_count'] = movie_stats['movie_rating_count'].fillna(0.0)
    
    mean_val_m = movie_stats['movie_mean_rating'].mean()
    std_val_m = movie_stats['movie_mean_rating'].std() + 1e-8
    count_mean_m = movie_stats['movie_rating_count'].mean()
    count_std_m = movie_stats['movie_rating_count'].std() + 1e-8
    
    movie_stats['movie_mean_rating_norm'] = (movie_stats['movie_mean_rating'] - mean_val_m) / std_val_m
    movie_stats['movie_rating_count_norm'] = (movie_stats['movie_rating_count'] - count_mean_m) / count_std_m
    
    movie_stats_dict = {}
    for _, row in movie_stats.iterrows():
        movie_stats_dict[int(row['movieId'])] = (row['movie_mean_rating_norm'], row['movie_rating_count_norm'])
        
    # Construction de la matrice movie_features (dim: num_movies, num_genres + 2)
    # Colonnes : [genre_0, genre_1, ..., genre_G-1, movie_mean, movie_count]
    movie_features_dim = num_genres + 2
    movie_features = np.zeros((num_movies, movie_features_dim), dtype=np.float32)
    
    for _, row in movies_df.iterrows():
        m_id = int(row['movieId'])
        if m_id in movie_to_idx:
            m_idx = movie_to_idx[m_id]
            
            # A. Multi-hot des genres
            genres_str = row.get('genres', '')
            if pd.notna(genres_str):
                for g in str(genres_str).split('|'):
                    g_clean = g.strip()
                    if g_clean in genre_to_col:
                        movie_features[m_idx, genre_to_col[g_clean]] = 1.0
            
            # B. Intégration des statistiques normées (par défaut 0.0)
            m_mean, m_count = movie_stats_dict.get(m_id, (0.0, 0.0))
            movie_features[m_idx, -2] = m_mean
            movie_features[m_idx, -1] = m_count
            
    # Sauvegarde des métadonnées pour l'inférence
    metadata = {
        "genres": genres_list,
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
        "global_mean_rating": float(user_mean_global)
    }
    
    # Création du dossier models s'il n'existe pas
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "twotower_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
        
    logging.info(f"Métadonnées et paramètres de normalisation sauvegardés dans {models_dir / 'twotower_metadata.json'}")
    
    return user_features, movie_features, user_features.shape[1], movie_features.shape[1]


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 256,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Charge les données, prépare les caractéristiques, effectue le split train/val
    et retourne les DataLoaders PyTorch associés.
    
    Args:
        data_dir: Path vers le dossier racine /data/ du projet.
        batch_size: Taille des batchs pour l'entraînement.
        val_split: Proportion des données pour la validation.
        random_seed: Graine pour la reproductibilité du split.
        
    Returns:
        Un tuple (train_loader, val_loader, metadata_dict)
    """
    raw_dir = data_dir / "raw" / "ml-latest-small"
    processed_dir = data_dir / "processed"
    
    # Chargement des ratings (requis)
    ratings_path = raw_dir / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings introuvables à : {ratings_path}")
    logging.info(f"Chargement des notes depuis {ratings_path}...")
    ratings_df = pd.read_csv(ratings_path)
    
    # Chargement des films (avec fallback sur le brut si le processed enrichi n'est pas dispo)
    movies_clean_path = processed_dir / "checkpoint_2_movies_clean.parquet"
    movies_raw_path = raw_dir / "movies.csv"
    
    if movies_clean_path.exists():
        logging.info(f"Chargement des films enrichis depuis {movies_clean_path}...")
        movies_df = pd.read_parquet(movies_clean_path)
    elif movies_raw_path.exists():
        logging.info(f"Fichier enrichi absent. Chargement des films bruts depuis {movies_raw_path}...")
        movies_df = pd.read_csv(movies_raw_path)
    else:
        raise FileNotFoundError("Aucun fichier de films trouvé.")
        
    # Création des index séquentiels uniques (requis par nn.Embedding)
    unique_users = sorted(ratings_df['userId'].unique())
    # Les films peuvent être tirés du fichier movies pour englober les films non notés
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
    
    # Division Train / Val
    logging.info("Division du dataset en Train et Validation...")
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(ratings_df))
    val_size = int(len(ratings_df) * val_split)
    
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]
    
    ratings_train = ratings_df.iloc[train_indices].copy()
    ratings_val = ratings_df.iloc[val_indices].copy()
    
    # Filtrer la validation pour s'assurer que les user/movie IDs existent dans nos mappings généraux
    # (Par construction c'est déjà le cas puisqu'on a pris les uniques sur tout le dataset)
    ratings_train = ratings_train[ratings_train['userId'].isin(user_to_idx) & ratings_train['movieId'].isin(movie_to_idx)]
    ratings_val = ratings_val[ratings_val['userId'].isin(user_to_idx) & ratings_val['movieId'].isin(movie_to_idx)]
    
    # Préparation des caractéristiques
    logging.info("Préparation des caractéristiques utilisateurs et films...")
    user_features, movie_features, u_dim, m_dim = prepare_features(
        ratings_train, movies_df, user_to_idx, movie_to_idx
    )
    
    # Instanciation des Datasets
    train_dataset = MovieLensDataset(ratings_train, user_features, movie_features, user_to_idx, movie_to_idx)
    val_dataset = MovieLensDataset(ratings_val, user_features, movie_features, user_to_idx, movie_to_idx)
    
    # Création des DataLoaders
    # shuffle=True pour le train afin de mélanger à chaque époque
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    metadata_dict = {
        "num_users": len(unique_users),
        "num_movies": len(unique_movies),
        "user_features_dim": u_dim,
        "movie_features_dim": m_dim,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx
    }
    
    logging.info(f"DataLoaders prêts. Train: {len(train_dataset)} échantillons, Val: {len(val_dataset)} échantillons.")
    logging.info(f"Nombre d'utilisateurs : {len(unique_users)}, Nombre de films : {len(unique_movies)}")
    
    return train_loader, val_loader, metadata_dict


if __name__ == "__main__":
    # Test unitaire rapide du loader
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "data"
    
    try:
        t_loader, v_loader, meta = get_dataloaders(DATA_PATH, batch_size=32)
        print("\n--- Test du Loader ---")
        print(f"User features dim: {meta['user_features_dim']}")
        print(f"Movie features dim: {meta['movie_features_dim']}")
        
        # Récupération d'un batch de test
        batch = next(iter(t_loader))
        print("Taille des tenseurs du batch :")
        for k, v in batch.items():
            print(f" - {k}: {v.shape} | type: {v.dtype}")
        print("✅ Le loader s'initialise et génère des batchs correctement !")
    except Exception as error:
        print(f"❌ Erreur lors du test du loader : {error}")
