import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from fastapi.testclient import TestClient
from src.v2_twotower.api.main import app

def test_api():
    with TestClient(app) as client:
        username = f"JeanCinema_{int(time.time())}"
        
        print("=== Étape 1 : Métadonnées API ===")
        res = client.get("/api/v2/ages")
        assert res.status_code == 200
        ages = res.json()
        
        res = client.get("/api/v2/occupations")
        assert res.status_code == 200
        occupations = res.json()
        
        assert "25" in ages
        assert "12" in occupations
        print(f"✅ Tranches d'âge ({len(ages)}) et professions ({len(occupations)}) vérifiées.")
        
        print(f"\n=== Étape 2 : Création de profil personnalisé ({username}) ===")
        signup_payload = {
            "username": username,
            "gender": "M",
            "age": 25,
            "occupation": 12,
            "zip_code": "75001",
            "ratings": [
                {"movieId": 1, "rating": 5.0}, # Toy Story -> Bien
                {"movieId": 2, "rating": 1.0}  # Jumanji -> Nul
            ],
            "top_n": 10
        }
        
        res = client.post("/api/v2/recommend", json=signup_payload)
        assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
        data = res.json()
        assert data.get("username") == username
        assert len(data.get("recommendations", [])) > 0
        print(f"✅ Création réussie, {len(data['recommendations'])} recommandations générées.")

        print("\n=== Étape 3 : Récupération du profil sauvegardé ===")
        res = client.get(f"/api/v2/users/{username}")
        assert res.status_code == 200
        user_data = res.json()
        assert user_data["username"] == username
        assert user_data["gender"] == "M"
        assert user_data["age"] == 25
        assert user_data["occupation"] == 12
        assert len(user_data["ratings"]) == 2
        ratings_dict = {r["movieId"]: r["rating"] for r in user_data["ratings"]}
        assert ratings_dict[1] == 5.0
        assert ratings_dict[2] == 1.0
        print("✅ Profil et notes SQLite restaurés avec succès.")

        print("\n=== Étape 4 : Reconnexion et recalcul des recommandations ===")
        login_payload = {
            "username": username,
            "ratings": [],
            "top_n": 10
        }
        res = client.post("/api/v2/recommend", json=login_payload)
        assert res.status_code == 200
        login_data = res.json()
        assert login_data.get("username") == username
        assert len(login_data.get("ratings", [])) == 2
        print(f"✅ Reconnexion réussie pour {username}.")

        print("\n=== Étape 5 : Feedback direct en session ===")
        first_rec_id = login_data["recommendations"][0]["movieId"]
        first_rec_title = login_data["recommendations"][0]["title"]
        print(f"Notation du film recommandé '{first_rec_title}' (ID {first_rec_id}) en 'Bien' (5.0)")
        
        feedback_payload = {
            "username": username,
            "ratings": [
                {"movieId": first_rec_id, "rating": 5.0}
            ],
            "top_n": 10
        }
        res = client.post("/api/v2/recommend", json=feedback_payload)
        assert res.status_code == 200
        feedback_data = res.json()
        assert len(feedback_data.get("ratings", [])) == 3
        recs_after_feedback = [r["movieId"] for r in feedback_data["recommendations"]]
        assert first_rec_id not in recs_after_feedback
        print("✅ Feedback pris en compte, recommandation exclut le film déjà noté.")

        print("\n=== Étape 6 : Persistance finale SQLite ===")
        res = client.get(f"/api/v2/users/{username}")
        assert res.status_code == 200
        final_user_data = res.json()
        assert len(final_user_data["ratings"]) == 3
        ratings_dict = {r["movieId"]: r["rating"] for r in final_user_data["ratings"]}
        assert ratings_dict[1] == 5.0
        assert ratings_dict[2] == 1.0
        assert ratings_dict[first_rec_id] == 5.0
        print("✅ Notes sauvegardées en base SQLite.")

        print("\n=== Étape 7 : Commentaires de la communauté ===")
        comment_payload = {
            "username": username,
            "comment_text": "Un chef d'oeuvre absolu de l'animation !"
        }
        res = client.post("/api/v2/movies/1/comments", json=comment_payload)
        assert res.status_code == 200
        comment_data = res.json()
        assert comment_data["username"] == username
        assert comment_data["comment_text"] == "Un chef d'oeuvre absolu de l'animation !"

        res = client.get("/api/v2/movies/1")
        assert res.status_code == 200
        movie_details = res.json()
        comments_list = movie_details.get("comments", [])
        assert any(c["username"] == username and c["comment_text"] == "Un chef d'oeuvre absolu de l'animation !" for c in comments_list)
        print("✅ Système de commentaires validé !")

if __name__ == "__main__":
    test_api()
