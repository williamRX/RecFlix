import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.v2_twotower.inference.recommender import TwoTowerRecommender
from src.v2_twotower.api.db import CommentsDB

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="RecFlix Two-Tower API",
    description="API de recommandation profonde utilisant l'architecture Two-Tower et l'accélération Apple Silicon."
)

# Global variables for caching
recommender: Optional[TwoTowerRecommender] = None
db: Optional[CommentsDB] = None
movies_list: List[Dict[str, Any]] = []
movies_dict: Dict[int, Dict[str, Any]] = {}
popular_movies: List[Dict[str, Any]] = []

# Pydantic models
class RatedMovie(BaseModel):
    movieId: int
    rating: float # 1.0 (Nul), 3.0 (Moyen), 5.0 (Bien)

class RecommendRequest(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None
    gender: Optional[str] = "M"
    age: Optional[int] = 25
    occupation: Optional[int] = 0
    zip_code: Optional[str] = "00000"
    liked_movie_ids: Optional[List[int]] = []
    ratings: Optional[List[RatedMovie]] = []
    top_n: int = 10


@app.on_event("startup")
def startup_event():
    """
    Initialise le moteur de recommandation et pré-charge les données de films en mémoire.
    """
    global recommender, movies_list, movies_dict, popular_movies, db
    logging.info("Démarrage du serveur : Initialisation du moteur Two-Tower...")
    
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        # Initialisation de la base de données de commentaires
        db = CommentsDB(db_path=data_dir / "processed" / "comments.db")
        logging.info("Base de données SQLite des commentaires initialisée avec succès.")

        # Initialisation du moteur d'inférence
        recommender = TwoTowerRecommender(data_dir=data_dir, models_dir=models_dir, device=device)
        
        # Chargement des détails de films enrichis (TMDB)
        enriched_path = data_dir / "processed" / "enriched_movies.csv"
        enriched_details = {}
        if enriched_path.exists():
            try:
                enriched_df = pd.read_csv(enriched_path)
                for _, row in enriched_df.iterrows():
                    m_id = int(row['movieId'])
                    enriched_details[m_id] = {
                        "poster_path": str(row['poster_path']) if pd.notna(row['poster_path']) else None,
                        "overview": str(row['overview']) if pd.notna(row['overview']) else "Aucun résumé disponible.",
                        "director": str(row['director']) if pd.notna(row['director']) else "Inconnu",
                        "top_3_cast": str(row['top_3_cast']) if pd.notna(row['top_3_cast']) else "Inconnu"
                    }
                logging.info(f"Chargement réussi de {len(enriched_details)} détails de films depuis {enriched_path.name}")
            except Exception as e:
                logging.error(f"Erreur lors du chargement des détails de films enrichis : {e}")
        else:
            logging.warning(f"Fichier de détails enrichis introuvable à {enriched_path}")

        # Chargement et indexation de la base de films
        movies_raw_path = data_dir / "raw" / "ml-1m" / "movies.dat"
        movies_meta_df = pd.read_csv(
            movies_raw_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="ISO-8859-1"
        )
        
        movies_list = [
            {
                "movieId": int(row['movieId']),
                "title": str(row['title']),
                "genres": str(row['genres']),
                "poster_path": enriched_details.get(int(row['movieId']), {}).get("poster_path"),
                "overview": enriched_details.get(int(row['movieId']), {}).get("overview", "Aucun résumé disponible."),
                "director": enriched_details.get(int(row['movieId']), {}).get("director", "Inconnu"),
                "top_3_cast": enriched_details.get(int(row['movieId']), {}).get("top_3_cast", "Inconnu")
            }
            for _, row in movies_meta_df.iterrows()
        ]
        
        movies_dict = {m["movieId"]: m for m in movies_list}
        
        # Identification des 24 films les plus populaires dans le train
        popular_movie_ids = recommender.popular_movie_ids
        popular_movies = [movies_dict[mid] for mid in popular_movie_ids if mid in movies_dict]
        
        logging.info(f"Serveur prêt ! {len(movies_list)} films indexés ({len(enriched_details)} avec détails), {len(popular_movies)} films populaires mis en cache.")
    except Exception as e:
        logging.error(f"Erreur fatale lors du démarrage du serveur : {e}", exc_info=True)


@app.get("/api/v2/occupations")
def get_occupations():
    """
    Retourne la liste des professions de MovieLens avec leurs libellés en français.
    """
    return {
        "0": "Autre ou non spécifié",
        "1": "Académique / Éducateur",
        "2": "Artiste / Métiers créatifs",
        "3": "Administrateur / Cadre administratif",
        "4": "Étudiant (Lycée/Université)",
        "5": "Métiers des services / Clientèle",
        "6": "Médecin / Professionnel de santé",
        "7": "Cadre supérieur / Directeur",
        "8": "Sans emploi",
        "9": "Écrivain / Journaliste",
        "10": "Métiers du bâtiment / Ouvrier",
        "11": "Avocat / Professionnel du droit",
        "12": "Programmeur / Développeur / Tech",
        "13": "Retraité",
        "14": "Vendeur / Commercial",
        "15": "Scientifique / Chercheur",
        "16": "Travailleur indépendant / Entrepreneur",
        "17": "Technicien",
        "18": "Artisan / Artisan d'art",
        "19": "Étudiant (Collège/Primaire)",
        "20": "Écrivain / Poète / Auteur"
    }


@app.get("/api/v2/ages")
def get_ages():
    """
    Retourne la liste des tranches d'âges MovieLens.
    """
    return {
        "1": "Moins de 18 ans",
        "18": "18-24 ans",
        "25": "25-34 ans",
        "35": "34-44 ans",
        "45": "45-49 ans",
        "50": "50-55 ans",
        "56": "56 ans et plus"
    }


@app.get("/api/v2/movies/search")
def search_movies(q: str):
    """
    Recherche des films par mot-clé dans le titre. Limité aux 15 premiers résultats.
    """
    if not q or len(q) < 2:
        return {"results": []}
    
    query = q.lower().strip()
    results = [
        m for m in movies_list 
        if query in m["title"].lower()
    ][:15]
    
    return {"results": results}


@app.get("/api/v2/movies/popular")
def get_popular_movies():
    """
    Retourne la liste des films populaires pour guider l'utilisateur lors du profilage.
    """
    return {"results": popular_movies}


@app.post("/api/v2/recommend")
def get_recommendations(request: RecommendRequest):
    """
    Calcule les recommandations du modèle Two-Tower V2 en fonction des critères envoyés.
    Supporte le mode existant (userId) ou cold-start/signup (données démo + films aimés).
    Enrichit les recommandations avec les nouvelles évaluations de session de l'utilisateur.
    """
    global recommender, movies_dict
    if recommender is None:
        raise HTTPException(status_code=500, detail="Le moteur Two-Tower n'est pas initialisé.")
        
    try:
        # Notes explicites soumises lors de la session
        session_ratings = [{"movieId": r.movieId, "rating": r.rating} for r in request.ratings] if request.ratings else []

        # Mode Utilisateur Existant (Warm Start MovieLens)
        if request.user_id is not None:
            if request.user_id not in recommender.user_to_idx:
                raise HTTPException(status_code=404, detail=f"L'utilisateur ID {request.user_id} n'existe pas dans la base d'entraînement.")
            
            # S'il n'y a pas de feedback explicite en session, on utilise l'inférence rapide par embedding précalculé
            if not session_ratings:
                recs = recommender.recommend_for_user(user_id=request.user_id, top_n=request.top_n)
            else:
                # Sinon on fusionne l'historique d'entraînement de l'utilisateur et ses avis de session
                user_idx = recommender.user_to_idx[request.user_id]
                u_demo = recommender._user_demographics.iloc[user_idx]
                gender = 'M' if u_demo['gender_idx'] == 1 else 'F'
                age = recommender.age_categories[int(u_demo['age_idx'])]
                occupation = int(u_demo['occupation_idx'])
                zip_idx = int(u_demo['zip_idx'])
                
                # Notes historiques du train set
                history_ratings = recommender.user_history_ratings.get(request.user_id, [])
                
                # Fusion des notations (le feedback de session écrase l'historique)
                ratings_map = {r["movieId"]: r["rating"] for r in history_ratings}
                for r in session_ratings:
                    ratings_map[r["movieId"]] = r["rating"]
                
                merged_ratings = [{"movieId": k, "rating": v} for k, v in ratings_map.items()]
                seen_movies = set(ratings_map.keys())
                
                recs = recommender.recommend_with_feedback(
                    gender=gender,
                    age=age,
                    occupation=occupation,
                    zip_code_or_idx=zip_idx,
                    ratings=merged_ratings,
                    seen_movie_ids=seen_movies,
                    top_n=request.top_n
                )
            
            # Enrichir avec poster_path et métadonnées
            for r in recs:
                m_id = r["movieId"]
                movie_meta = movies_dict.get(m_id, {})
                r["poster_path"] = movie_meta.get("poster_path")
                r["overview"] = movie_meta.get("overview")
                r["director"] = movie_meta.get("director")
                r["top_3_cast"] = movie_meta.get("top_3_cast")
            return {
                "mode": "warm_start",
                "user_id": request.user_id,
                "recommendations": recs
            }
            
        # Mode Utilisateur Personnalisé (Reconnexion / Inscription)
        elif request.username is not None:
            username = request.username.strip()
            existing_user = db.get_user(username)
            
            if existing_user is not None:
                # Utilisateur existant : on enregistre les nouvelles notes de session
                for r in session_ratings:
                    db.save_rating(username, r["movieId"], r["rating"])
                
                # Chargement de l'historique d'évaluation complet
                history_ratings = db.get_user_ratings(username)
                ratings_map = {r["movieId"]: r["rating"] for r in history_ratings}
                
                merged_ratings = [{"movieId": k, "rating": v} for k, v in ratings_map.items()]
                seen_movies = set(ratings_map.keys())
                
                recs = recommender.recommend_with_feedback(
                    gender=existing_user["gender"],
                    age=existing_user["age"],
                    occupation=existing_user["occupation"],
                    zip_code_or_idx=existing_user["zip_code"],
                    ratings=merged_ratings,
                    seen_movie_ids=seen_movies,
                    top_n=request.top_n
                )
                
                # Enrichir avec poster_path et métadonnées
                for r in recs:
                    m_id = r["movieId"]
                    movie_meta = movies_dict.get(m_id, {})
                    r["poster_path"] = movie_meta.get("poster_path")
                    r["overview"] = movie_meta.get("overview")
                    r["director"] = movie_meta.get("director")
                    r["top_3_cast"] = movie_meta.get("top_3_cast")
                    
                return {
                    "mode": "custom_user",
                    "username": username,
                    "gender": existing_user["gender"],
                    "age": existing_user["age"],
                    "occupation": existing_user["occupation"],
                    "zip_code": existing_user["zip_code"],
                    "ratings": merged_ratings,
                    "recommendations": recs
                }
            else:
                # Nouvel utilisateur : Enregistrement
                # S'assurer que les critères démographiques sont fournis
                if not request.gender or not request.age or request.occupation is None:
                    raise HTTPException(status_code=404, detail="Utilisateur introuvable. Veuillez d'abord vous inscrire.")
                
                # Sauvegarde du profil
                db.save_user(
                    username=username,
                    gender=request.gender,
                    age=request.age,
                    occupation=request.occupation,
                    zip_code=request.zip_code
                )
                
                # Sauvegarde des notes initiales
                for r in session_ratings:
                    db.save_rating(username, r["movieId"], r["rating"])
                
                recs = recommender.recommend_with_feedback(
                    gender=request.gender,
                    age=request.age,
                    occupation=request.occupation,
                    zip_code_or_idx=request.zip_code,
                    ratings=session_ratings,
                    seen_movie_ids=set(r["movieId"] for r in session_ratings),
                    top_n=request.top_n
                )
                
                # Enrichir avec poster_path et métadonnées
                for r in recs:
                    m_id = r["movieId"]
                    movie_meta = movies_dict.get(m_id, {})
                    r["poster_path"] = movie_meta.get("poster_path")
                    r["overview"] = movie_meta.get("overview")
                    r["director"] = movie_meta.get("director")
                    r["top_3_cast"] = movie_meta.get("top_3_cast")
                    
                return {
                    "mode": "custom_user",
                    "username": username,
                    "gender": request.gender,
                    "age": request.age,
                    "occupation": request.occupation,
                    "zip_code": request.zip_code,
                    "ratings": session_ratings,
                    "recommendations": recs
                }
        else:
            raise HTTPException(status_code=400, detail="Requête invalide. Veuillez fournir un user_id ou un username.")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Erreur d'inférence : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul des recommandations : {str(e)}")


# Pydantic models pour les commentaires
class CommentRequest(BaseModel):
    username: str
    comment_text: str


@app.get("/api/v2/movies/{movieId}")
def get_movie_details(movieId: int):
    """
    Retourne les détails complets d'un film et sa liste de commentaires de la communauté.
    """
    global movies_dict, db
    if movieId not in movies_dict:
        raise HTTPException(status_code=404, detail="Film introuvable.")
    
    movie = movies_dict[movieId].copy()
    movie["comments"] = db.get_comments(movieId) if db is not None else []
    return movie


@app.post("/api/v2/movies/{movieId}/comments")
def post_movie_comment(movieId: int, request: CommentRequest):
    """
    Publie un nouveau commentaire pour un film donné.
    """
    global db, movies_dict
    if movieId not in movies_dict:
        raise HTTPException(status_code=404, detail="Film introuvable.")
    if db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible.")
        
    try:
        new_comment = db.add_comment(
            movie_id=movieId,
            username=request.username,
            comment_text=request.comment_text
        )
        return new_comment
    except Exception as e:
        logging.error(f"Erreur lors de l'enregistrement du commentaire : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de base de données : {str(e)}")


@app.get("/api/v2/users/{username}")
def get_user_profile(username: str):
    """
    Récupère le profil d'un utilisateur custom et ses notations de la base SQLite.
    """
    global db
    if db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible.")
    
    user = db.get_user(username)
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
        
    ratings = db.get_user_ratings(username)
    return {
        "username": user["username"],
        "gender": user["gender"],
        "age": user["age"],
        "occupation": user["occupation"],
        "zip_code": user["zip_code"],
        "ratings": ratings
    }


# Montage du frontend statique (doit être fait en dernier)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
