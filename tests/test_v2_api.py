import requests
import sys
import time

def test_api():
    base_url = "http://localhost:8000"
    username = f"JeanCinema_{int(time.time())}"
    
    print("=== Étape 1 : Vérification de la disponibilité du serveur ===")
    try:
        res = requests.get(f"{base_url}/api/v2/ages")
        res.raise_for_status()
        print("✅ Serveur disponible.")
    except Exception as e:
        print(f"❌ Impossible de se connecter au serveur à {base_url} : {e}")
        sys.exit(1)
        
    print("\n=== Étape 2 : Vérification des métadonnées ===")
    ages = requests.get(f"{base_url}/api/v2/ages").json()
    occupations = requests.get(f"{base_url}/api/v2/occupations").json()
    print(f"Tranches d'âge disponibles : {len(ages)}")
    print(f"Professions disponibles : {len(occupations)}")
    
    # 25 ans corresponds à la clé "25"
    # Développeur/Programmeur corresponds à la clé "12"
    assert "25" in ages
    assert "12" in occupations
    
    print(f"\n=== Étape 3 : Création d'un nouveau profil personnalisé ({username}) ===")
    # On simule le Cold Start en envoyant les métadonnées démographiques et deux films notés :
    # Toy Story (movieId=1) : Bien (5.0)
    # Jumanji (movieId=2) : Nul (1.0)
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
    
    res = requests.post(f"{base_url}/api/v2/recommend", json=signup_payload)
    if res.status_code == 200:
        data = res.json()
        print("✅ Création et calcul des recommandations réussis.")
        print(f"Mode retourné : {data.get('mode')}")
        print(f"Pseudo retourné : {data.get('username')}")
        print(f"Nombre de recommandations obtenues : {len(data.get('recommendations', []))}")
        assert data.get('username') == username
        assert len(data.get('recommendations', [])) > 0
    else:
        print(f"❌ Échec de la création : {res.status_code} - {res.text}")
        sys.exit(1)

    print("\n=== Étape 4 : Récupération du profil sauvegardé ===")
    res = requests.get(f"{base_url}/api/v2/users/{username}")
    if res.status_code == 200:
        user_data = res.json()
        print("✅ Profil récupéré avec succès.")
        print(f"Données : {user_data}")
        assert user_data["username"] == username
        assert user_data["gender"] == "M"
        assert user_data["age"] == 25
        assert user_data["occupation"] == 12
        assert len(user_data["ratings"]) == 2
        # Vérification des notes
        ratings_dict = {r["movieId"]: r["rating"] for r in user_data["ratings"]}
        assert ratings_dict[1] == 5.0
        assert ratings_dict[2] == 1.0
    else:
        print(f"❌ Échec de la récupération du profil : {res.status_code} - {res.text}")
        sys.exit(1)

    print("\n=== Étape 5 : Reconnexion et recalcul des recommandations ===")
    # On simule une reconnexion : on envoie juste le username et les ratings actuels (vides au départ, 
    # mais l'API charge l'historique de la base).
    login_payload = {
        "username": username,
        "ratings": [],
        "top_n": 10
    }
    res = requests.post(f"{base_url}/api/v2/recommend", json=login_payload)
    if res.status_code == 200:
        login_data = res.json()
        print("✅ Reconnexion réussie.")
        print(f"Profil : {login_data.get('gender')}, {login_data.get('age')} ans, profession {login_data.get('occupation')}")
        print(f"Notes historiques restaurées : {len(login_data.get('ratings', []))}")
        assert login_data.get("username") == username
        assert len(login_data.get("ratings", [])) == 2
        
        # On sauvegarde les recommandations initiales pour comparaison
        recs_initial = [r["movieId"] for r in login_data["recommendations"]]
        print(f"Top 5 recommandations initiales : {recs_initial[:5]}")
    else:
        print(f"❌ Échec de la reconnexion : {res.status_code} - {res.text}")
        sys.exit(1)

    print("\n=== Étape 6 : Ajout de feedback en direct ===")
    # On va voter pour un autre film, par exemple le premier recommandé de la liste
    first_rec_id = login_data["recommendations"][0]["movieId"]
    first_rec_title = login_data["recommendations"][0]["title"]
    print(f"Notation du film recommandé '{first_rec_title}' (ID {first_rec_id}) en 'Bien' (5.0)")
    
    feedback_payload = {
        "username": username,
        # On envoie la liste des notes en incluant le nouveau feedback
        "ratings": [
            {"movieId": first_rec_id, "rating": 5.0}
        ],
        "top_n": 10
    }
    res = requests.post(f"{base_url}/api/v2/recommend", json=feedback_payload)
    if res.status_code == 200:
        feedback_data = res.json()
        print("✅ Feedback envoyé et recommandations recalculées.")
        print(f"Nouvel historique de notes : {len(feedback_data.get('ratings', []))}")
        assert len(feedback_data.get('ratings', [])) == 3
        
        recs_after_feedback = [r["movieId"] for r in feedback_data["recommendations"]]
        print(f"Top 5 recommandations après feedback : {recs_after_feedback[:5]}")
        # Le film noté ne doit plus apparaître dans les recommandations puisqu'il est marqué comme "vu/noté"
        assert first_rec_id not in recs_after_feedback
    else:
        print(f"❌ Échec du feedback : {res.status_code} - {res.text}")
        sys.exit(1)

    print("\n=== Étape 7 : Vérification finale de la persistance de l'historique après déconnexion ===")
    # On se reconnecte une nouvelle fois sans préciser de nouveaux votes
    res = requests.get(f"{base_url}/api/v2/users/{username}")
    if res.status_code == 200:
        final_user_data = res.json()
        print(f"Nombre final de notes en base : {len(final_user_data['ratings'])}")
        assert len(final_user_data["ratings"]) == 3
        ratings_dict = {r["movieId"]: r["rating"] for r in final_user_data["ratings"]}
        assert ratings_dict[1] == 5.0
        assert ratings_dict[2] == 1.0
        assert ratings_dict[first_rec_id] == 5.0
        print("✅ Tout fonctionne parfaitement ! La persistance, le recalcul et le feedback en direct sont validés.")
    else:
        print(f"❌ Échec de la vérification finale : {res.status_code}")
        sys.exit(1)

    print("\n=== Étape 8 : Test du système de commentaires de la communauté ===")
    # On ajoute un commentaire pour le film 1
    comment_payload = {
        "username": username,
        "comment_text": "Un chef d'oeuvre absolu de l'animation !"
    }
    res = requests.post(f"{base_url}/api/v2/movies/1/comments", json=comment_payload)
    if res.status_code == 200:
        comment_data = res.json()
        print("✅ Commentaire ajouté avec succès.")
        print(f"Détails du commentaire : {comment_data}")
        assert comment_data["username"] == username
        assert comment_data["comment_text"] == "Un chef d'oeuvre absolu de l'animation !"
    else:
        print(f"❌ Échec de l'ajout du commentaire : {res.status_code} - {res.text}")
        sys.exit(1)

    # Récupérer les commentaires du film 1
    res = requests.get(f"{base_url}/api/v2/movies/1")
    if res.status_code == 200:
        movie_details = res.json()
        comments_list = movie_details.get("comments", [])
        print(f"Commentaires trouvés sur Toy Story : {len(comments_list)}")
        assert len(comments_list) > 0
        assert any(c["username"] == username and c["comment_text"] == "Un chef d'oeuvre absolu de l'animation !" for c in comments_list)
        print("✅ Le système de commentaires fonctionne parfaitement.")
    else:
        print(f"❌ Échec de la récupération des détails du film : {res.status_code}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
