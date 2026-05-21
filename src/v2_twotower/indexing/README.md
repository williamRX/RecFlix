# Indexation Vectorielle pour la Recherche de Plus Proches Voisins (ANN)

Ce répertoire contiendra la logique d'indexation des embeddings de films. L'indexation est cruciale dans les architectures Two-Tower pour permettre une récupération ultra-rapide des meilleurs films (étape de *Retrieval*) parmi des millions de candidats.

## Contenu attendu

1. **`build_index.py`** :
   - Charge la matrice d'embeddings de films extraite.
   - Construit un index de recherche approximative de plus proches voisins (ANN).
   - Outils recommandés :
     - **FAISS** (Facebook AI Similarity Search) pour la performance.
     - **Scann** ou **HNSWLib**.
     - Une implémentation Numpy simple (`IndexFlatIP` ou calcul de similarité cosinus matriciel) pour le dataset MovieLens Small.
   - Sauvegarde l'index construit sur le disque (`movie_index.faiss` ou similaire).
