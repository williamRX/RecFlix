from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from pathlib import Path
import logging

from src.v1_baseline.inference.recommender import ProfileRecommender

# Configuration
logging.basicConfig(level=logging.INFO)

# Initialisation de l'application
app = FastAPI(title="RecFlix API", description="Moteur de recommandation en 3D")

# Singleton du recommender pour éviter de recharger la matrice à chaque requête
recommender = None
map_3d_data = None
map_2d_data = None

# Modèles Pydantic pour la validation des requêtes
class RatingInput(BaseModel):
    movieId: int
    rating: float

class RecommendationRequest(BaseModel):
    user_ratings: List[RatingInput]
    top_n: int = 10

@app.on_event("startup")
def startup_event():
    global recommender, map_3d_data, map_2d_data
    logging.info("Démarrage de l'API : Chargement du modèle SVD...")
    try:
        recommender = ProfileRecommender()
        
        # Chargement en cache de la carte 3D
        project_root = Path(__file__).resolve().parents[3]
        map_path = project_root / "data" / "processed" / "movies_3d_map.parquet"
        if map_path.exists():
            df_map = pd.read_parquet(map_path)
            # On le stocke en mémoire formaté pour le JSON
            map_3d_data = df_map.to_dict(orient="records")
            logging.info(f"Carte 3D chargée ({len(map_3d_data)} films).")
        else:
            logging.warning("Fichier movies_3d_map.parquet introuvable.")
            
        # Chargement de la carte 2D
        map2d_path = project_root / "data" / "processed" / "movies_2d_map.parquet"
        if map2d_path.exists():
            df_map2d = pd.read_parquet(map2d_path)
            map_2d_data = df_map2d.to_dict(orient="records")
            logging.info(f"Carte 2D chargée ({len(map_2d_data)} films).")
        else:
            logging.warning("Fichier movies_2d_map.parquet introuvable.")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation : {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/map3d")
def get_map_3d():
    """Retourne les coordonnées 3D (x, y, z) de tous les films pour Plotly."""
    if map_3d_data is None:
        raise HTTPException(status_code=404, detail="Carte 3D non générée")
    return {"points": map_3d_data}

@app.get("/api/map2d")
def get_map_2d():
    """Retourne les coordonnées 2D (x, y) de tous les films pour Plotly."""
    if map_2d_data is None:
        raise HTTPException(status_code=404, detail="Carte 2D non générée")
    return {"points": map_2d_data}

@app.post("/api/recommend")
def get_recommendations(request: RecommendationRequest):
    """Calcule les recommandations basées sur l'espace latent."""
    if recommender is None:
        raise HTTPException(status_code=500, detail="Moteur non initialisé")
        
    ratings = [{"movieId": r.movieId, "rating": r.rating} for r in request.user_ratings]
    recs = recommender.get_recommendations(user_ratings=ratings, top_n=request.top_n)
    return {"recommendations": recs}

# Montage du frontend à la racine. 
# IMPORTANT : Toujours faire les app.mount() EN DERNIER, sinon ils interceptent les autres routes.
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
