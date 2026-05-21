# API de Service (FastAPI) pour Two-Tower V2

Ce répertoire contiendra le code permettant d'exposer le modèle de recommandation Two-Tower V2 sous forme d'API REST.

## Contenu attendu

1. **`main.py`** :
   - Initialise une application **FastAPI** dédiée à la V2.
   - Charge en mémoire (au démarrage via les évènements de startup) :
     - Les poids du modèle Two-Tower (`twotower_best_model.pth`).
     - L'index vectoriel de films pré-calculé (`movie_index.faiss`).
     - Les dictionnaires de mapping (`twotower_mappings.json`).
     - Les métadonnées de normalisation (`twotower_metadata.json`).
   - Expose des points d'accès (endpoints) comme :
     - `/api/v2/recommend` : prend en entrée un `userId` ou une liste de notes dynamiques et renvoie les top-N recommandations.
     - `/api/v2/embeddings` : permet de récupérer la représentation vectorielle d'un film ou d'un utilisateur.
