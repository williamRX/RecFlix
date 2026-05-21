# Moteur d'Inférence (Retrieval & Ranking)

Ce répertoire contiendra la logique d'inférence en production. Il orchestre la génération de recommandations personnalisées pour un utilisateur.

## Contenu attendu

1. **`recommender.py`** :
   - Reçoit une requête contenant l'identifiant d'un utilisateur (`userId`) ou une liste de ses notes récentes (pour l'inférence *cold-start* ou à la volée).
   - Construit le vecteur de profil de l'utilisateur :
     - Si l'utilisateur est connu : passe son ID et ses métadonnées précalculées dans la **Tour Utilisateur** (`UserTower`).
     - Si l'utilisateur est nouveau/dynamique : extrait son vecteur d'embedding basé sur ses notations à la volée.
   - Interroge l'index vectoriel de films (`movie_index.faiss`) pour récupérer les $K$ films les plus proches (similarité cosinus maximale).
   - Filtre les films que l'utilisateur a déjà vus.
   - Retourne la liste finale des recommandations triées avec leurs scores.
