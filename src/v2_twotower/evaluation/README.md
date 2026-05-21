# Évaluation Off-line et Métriques de Classement

Ce répertoire contient les outils pour évaluer précisément la qualité de recommandation du modèle Two-Tower V2, au-delà de la perte MSE globale.

## Contenu attendu

1. **`metrics.py`** :
   - Calcule des métriques de classement standard de l'état de l'art (sur l'ensemble de validation) :
     - **NDCG@K** (Normalized Discounted Cumulative Gain) : évalue la pertinence de l'ordre des recommandations.
     - **Recall@K** et **Precision@K** : mesurent la capacité à retrouver les films notés positivement par l'utilisateur.
     - **MAP@K** (Mean Average Precision).
     - **Hit Rate@K** (Taux de succès).
     
2. **`evaluator.py`** :
   - Permet de comparer les performances du modèle Two-Tower V2 par rapport aux baselines (KNN Cosine V1 et SVD V1).
   - Génère un rapport de performance structuré et des courbes de métriques.
