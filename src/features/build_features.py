import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 4. Définition des Poids (Paramétrables)
# ==========================================
# Ces poids permettent d'ajuster l'importance de chaque feature dans le calcul
# de similarité finale. Par exemple, avoir le même réalisateur aura plus 
# d'impact que d'avoir la même décennie.
WEIGHT_GENRE = 3.0
WEIGHT_DIRECTOR = 2.5
WEIGHT_CAST = 2.0
WEIGHT_TAGS = 1.5
WEIGHT_OVERVIEW = 1.0
WEIGHT_DECADE = 0.5

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-latest-small"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

def build_features():
    """Script principal de création de la matrice de features."""
    
    # Création du dossier processed s'il n'existe pas
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. Ingestion et Pré-traitement
    # ==========================================
    logging.info("Étape 1 : Ingestion et Pré-traitement")
    try:
        movies = pd.read_csv(RAW_DIR / "movies.csv")
        tags = pd.read_csv(RAW_DIR / "tags.csv")
        # On ne garde que les colonnes utiles de enriched pour éviter les doublons lors du merge
        enriched = pd.read_csv(PROC_DIR / "enriched_movies.csv")[['movieId', 'director', 'top_3_cast', 'overview', 'poster_path']]
    except Exception as e:
        logging.error(f"Erreur lors du chargement des fichiers : {e}")
        return

    # Grouper les tags par movieId
    tags['tag'] = tags['tag'].astype(str)
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Extraire la décennie depuis le titre (ex: "(1995)" -> 1990s)
    years = movies['title'].str.extract(r'\((\d{4})\)')
    movies['decade'] = years[0].astype(float) // 10 * 10
    # Remplacer les valeurs nulles par 0 (pour les films sans année) et formatter
    movies['decade'] = movies['decade'].fillna(0).astype(int).astype(str) + "s"
    
    logging.info("Sauvegarde du CHECKPOINT 1...")
    movies.to_parquet(PROC_DIR / "checkpoint_1_movies.parquet", index=False)
    tags_grouped.to_parquet(PROC_DIR / "checkpoint_1_tags.parquet", index=False)

    # ==========================================
    # 2. Fusion et Nettoyage
    # ==========================================
    logging.info("Étape 2 : Fusion et Nettoyage")
    # Merge global
    df = movies.merge(enriched, on='movieId', how='left')
    df = df.merge(tags_grouped, on='movieId', how='left')
    
    # Remplacer les NaN des tags par du vide (car optionnels)
    df['tag'] = df['tag'].fillna("")
    
    # Dropna strict demandé sur les colonnes clés
    df_clean = df.dropna(subset=['overview', 'director', 'top_3_cast']).copy()
    df_clean = df_clean.reset_index(drop=True) # Reset de l'index indispensable après un dropna
    
    logging.info(f"Dimensions après nettoyage : {df_clean.shape[0]} films conservés.")
    
    logging.info("Sauvegarde du CHECKPOINT 2...")
    df_clean.to_parquet(PROC_DIR / "checkpoint_2_movies_clean.parquet", index=False)

    # ==========================================
    # 3. Feature Engineering
    # ==========================================
    logging.info("Étape 3 : Feature Engineering")
    
    try:
        # A. Genres (Multi-Hot -> L2)
        # On utilise CountVectorizer avec binary=True pour faire un One-Hot Encoding multiple
        cv_genre = CountVectorizer(tokenizer=lambda x: x.split('|'), binary=True, token_pattern=None)
        genres_matrix = cv_genre.fit_transform(df_clean['genres'])
        genres_matrix = normalize(genres_matrix, norm='l2')
        
        # B. Decade (One-Hot -> L2)
        cv_decade = CountVectorizer(binary=True)
        decade_matrix = cv_decade.fit_transform(df_clean['decade'])
        decade_matrix = normalize(decade_matrix, norm='l2')
        
        # C. Director (Prefixe -> One-Hot -> L2)
        def clean_director(d):
            return "dir_" + str(d).replace(" ", "").replace("-", "")
            
        df_clean['director_clean'] = df_clean['director'].apply(clean_director)
        cv_director = CountVectorizer(binary=True)
        director_matrix = cv_director.fit_transform(df_clean['director_clean'])
        director_matrix = normalize(director_matrix, norm='l2')
        
        # D. Cast (Prefixe -> Multi-Hot -> L2)
        def clean_cast(c):
            actors = str(c).split('|')
            return " ".join(["cast_" + a.replace(" ", "").replace("-", "") for a in actors])
            
        df_clean['cast_clean'] = df_clean['top_3_cast'].apply(clean_cast)
        cv_cast = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b") # regex pour accepter les préfixes "cast_"
        cast_matrix = cv_cast.fit_transform(df_clean['cast_clean'])
        cast_matrix = normalize(cast_matrix, norm='l2')
        
        # E. Tags (TF-IDF)
        tfidf_tags = TfidfVectorizer(stop_words='english')
        tags_matrix = tfidf_tags.fit_transform(df_clean['tag'])
        # Le TF-IDF applique déjà une normalisation L2 par défaut
        
        # F. Overview (TF-IDF)
        tfidf_overview = TfidfVectorizer(stop_words='english', max_features=5000)
        overview_matrix = tfidf_overview.fit_transform(df_clean['overview'])
        
    except Exception as e:
        logging.error(f"Erreur durant la création des features : {e}")
        return

    # ==========================================
    # 4. Pondération et Assemblage
    # ==========================================
    logging.info("Étape 4 : Pondération et Assemblage (scipy.sparse.hstack)")
    
    genres_w = genres_matrix * WEIGHT_GENRE
    decade_w = decade_matrix * WEIGHT_DECADE
    director_w = director_matrix * WEIGHT_DIRECTOR
    cast_w = cast_matrix * WEIGHT_CAST
    tags_w = tags_matrix * WEIGHT_TAGS
    overview_w = overview_matrix * WEIGHT_OVERVIEW
    
    # Assemblage horizontal des matrices creuses
    feature_matrix = sp.hstack([genres_w, decade_w, director_w, cast_w, tags_w, overview_w], format='csr')
    
    logging.info(f"==> Dimensions de la Matrice de Features finale : {feature_matrix.shape}")

    # ==========================================
    # 5. Export de la Matrice et du Vocabulaire
    # ==========================================
    logging.info("Étape 5 : Export au format npz et json")
    try:
        export_path = PROC_DIR / "features_matrix.npz"
        sp.save_npz(export_path, feature_matrix)
        logging.info(f"Succès ! Matrice sauvegardée dans : {export_path}")
        
        # Récupération des noms de colonnes depuis les vectorizers
        import json
        vocab = (
            list(cv_genre.get_feature_names_out()) +
            list(cv_decade.get_feature_names_out()) +
            list(cv_director.get_feature_names_out()) +
            list(cv_cast.get_feature_names_out()) +
            list(tfidf_tags.get_feature_names_out()) +
            list(tfidf_overview.get_feature_names_out())
        )
        
        vocab_path = PROC_DIR / "feature_names.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        logging.info(f"Succès ! Vocabulaire sauvegardé dans : {vocab_path}")
        
    except Exception as e:
        logging.error(f"Erreur lors de l'export : {e}")

if __name__ == "__main__":
    build_features()
