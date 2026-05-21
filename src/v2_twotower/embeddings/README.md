# Extraction et Visualisation des Embeddings (Espace Latent)

Ce répertoire contiendra les scripts permettant d'extraire, de stocker et d'analyser les représentations vectorielles (embeddings) générées par le modèle Two-Tower une fois l'entraînement terminé.

## Contenu attendu

1. **`extractor.py`** :
   - Charge les poids entraînés du modèle (`twotower_best_model.pth`).
   - Passe l'ensemble des films dans la **Tour Item** (`ItemTower`) pour obtenir une représentation vectorielle de taille $D$ (ex. 32) pour chaque film.
   - Enregistre la matrice d'embeddings sous forme de fichier numpy (`movie_embeddings.npy`) ou format sérialisé léger (ex: Parquet/Feather).
   
2. **`visualizer.py`** :
   - Charge les embeddings des films et réduit la dimensionnalité (en 2D ou 3D) à l'aide d'algorithmes comme **t-SNE**, **UMAP** ou **PCA**.
   - Génère des graphiques interactifs (via Plotly) pour observer les regroupements (clusters) de films par genre, année ou popularité.
