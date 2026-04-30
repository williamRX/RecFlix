# RecFlix - Système de Recommandation de Films

Ce document regroupe toutes les commandes utiles pour lancer et travailler sur le projet **RecFlix** en utilisant Docker. L'utilisation de Docker te permet de ne pas te soucier des versions de Python installées sur ta machine locale.

## 🚀 Démarrer l'environnement

Pour construire l'image et lancer l'API en tâche de fond :

```bash
# Construit l'image Docker (à faire lors de la première utilisation ou si tu modifies requirements.txt / Dockerfile)
docker-compose build

# Lance le conteneur en arrière-plan
docker-compose up -d

# Voir les logs en direct (pour vérifier que l'API a bien démarré)
docker-compose logs -f
```

Une fois lancé, tu peux accéder à ton API FastAPI via ton navigateur :
- **L'API :** [http://localhost:8000](http://localhost:8000)
- **La documentation interactive (Swagger UI) :** [http://localhost:8000/docs](http://localhost:8000/docs)

## 📦 Télécharger le Dataset MovieLens

Puisque tu n'as pas Python configuré globalement sur ta machine locale, la meilleure pratique est d'exécuter tes scripts directement **à l'intérieur du conteneur Docker**. 

Comme les dossiers `src` et `data` sont synchronisés (grâce aux volumes dans `docker-compose.yml`), les fichiers téléchargés par le conteneur apparaîtront directement sur ton Mac !

Pour télécharger le dataset :

```bash
# Exécute le script Python dans un conteneur éphémère
docker-compose run --rm api python src/data/download_dataset.py
```
*Note : l'option `--rm` supprime le conteneur une fois le script terminé pour garder ton système propre.*

## 💻 Exécuter d'autres commandes ou scripts

Si tu as besoin de lancer d'autres scripts ou d'entrer dans le conteneur pour explorer :

```bash
# Lancer un autre script (ex: un futur script de transformation)
docker-compose run --rm api python src/features/ton_script.py

# Ouvrir un terminal interactif (bash) à l'intérieur du conteneur
docker-compose run --rm api bash

# Installer une nouvelle dépendance (si tu es dans le bash du conteneur)
pip install nom_du_package
# N'oublie pas de l'ajouter dans requirements.txt ensuite !
```

## 🛑 Arrêter l'environnement

Quand tu as fini de travailler :

```bash
# Arrête les conteneurs sans les supprimer
docker-compose stop

# Arrête et supprime les conteneurs (recommandé pour repartir au propre)
docker-compose down
```
