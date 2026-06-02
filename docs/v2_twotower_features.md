# Documentation des Caractéristiques (Features) - Architecture Two-Tower

Ce document détaille toutes les caractéristiques (démographiques, temporelles et comportementales) utilisées par les deux tours de notre modèle de recommandation profond V2.

---

## 👤 1. Tour Utilisateur (User Tower)

La Tour Utilisateur fusionne l'identité, les données sociodémographiques de l'utilisateur et son profil comportemental calculé sur l'historique des notes d'entraînement.

### A. Données d'Identité & Démographiques (Entrées Catégorielles)

Ces caractéristiques passent par des tables d'embeddings individuelles qui apprennent des représentations vectorielles denses pour chaque catégorie :

1. **`userId` (ID Utilisateur)**
   - *Source* : `ratings.dat` et `users.dat`.
   - *Cardinalité* : 6 040 utilisateurs uniques.
   - *Traitement* : Indexation de $0$ à $6039$.
   - *Dimension de l'embedding* : 32.

2. **`gender` (Genre)**
   - *Source* : `users.dat` (colonne 2).
   - *Format* : `'M'` (Homme) ou `'F'` (Femme).
   - *Traitement* : Encodage binaire (0 pour 'F', 1 pour 'M').
   - *Dimension de l'embedding* : 2.

3. **`age` (Tranche d'âge)**
   - *Source* : `users.dat` (colonne 3).
   - *Format* : Catégories codées par des entiers :
     - `1` : Moins de 18 ans
     - `18` : 18-24 ans
     - `25` : 25-34 ans
     - `35` : 35-44 ans
     - `45` : 45-49 ans
     - `50` : 50-55 ans
     - `56` : 56 ans et plus
   - *Traitement* : Mapping vers des index séquentiels de $0$ à $6$.
   - *Dimension de l'embedding* : 4.

4. **`occupation` (Profession)**
   - *Source* : `users.dat` (colonne 4).
   - *Format* : Entiers de $0$ à $20$ représentant 21 catégories de métiers (ex: `4` pour "college/grad student", `12` pour "programmer").
   - *Traitement* : Déjà séquentiel (aucune transformation nécessaire).
   - *Dimension de l'embedding* : 8.

5. **`zip_code` (Zone géographique)**
   - *Source* : `users.dat` (colonne 5).
   - *Format* : Code postal américain (ex: `48067`).
   - *Traitement* : Conservation du **premier chiffre uniquement** (représente l'une des 10 grandes régions postales des États-Unis, de `'0'` à `'9'`). Les valeurs invalides sont remplacées par `'0'`.
   - *Dimension de l'embedding* : 4.

---

### B. Caractéristiques Comportementales (Entrées Numériques)

Ces caractéristiques décrivent l'activité et les goûts de l'utilisateur. Elles sont regroupées dans un tenseur continu de dimension **48** :

6. **Statistiques Globales (`user_numeric` : dim 2)**
   - **Activité (`user_rating_count`)** : Nombre total de films notés par l'utilisateur (normalisé par Z-score).
   - **Générosité (`user_mean_rating`)** : Note moyenne de l'utilisateur (normalisée par Z-score).

7. **Profil d'Intérêt Thématique (`genre_proportions` : dim 18)**
   - *Calcul* : Ratio de films vus par genre.
   - *Exemple* : Si l'utilisateur a noté 10 films d'Action sur 20 notes au total, sa valeur pour l'Action est `0.5`.
   - *Format* : 18 valeurs continues dans l'intervalle $[0.0, 1.0]$. La somme de ces proportions est égale à 1.0.

8. **Profil d'Appréciation Relative (`genre_satisfactions` : dim 18)**
   - *Calcul* : Écart entre la note moyenne de l'utilisateur pour le genre $G$ et sa note moyenne globale :
     $$\text{satisfaction}_G = \text{moyenne}_G - \text{moyenne}_{\text{globale}}$$
   - *Gestion des valeurs manquantes* : Si l'utilisateur n'a jamais noté le genre $G$, la valeur est de `0.0` (effet neutre).
   - *Format* : 18 valeurs continues (valeurs positives = aime plus que sa moyenne, négatives = aime moins).

9. **Profil d'Affinité Temporelle (`decade_proportions` : dim 10)**
   - *Calcul* : Ratio de films notés par décennie de sortie (10 décennies disponibles de 1910s à 2000s).
   - *Format* : 10 valeurs continues dans l'intervalle $[0.0, 1.0]$.

---

## 🎬 2. Tour Film (Item Tower)

La Tour Film encode les métadonnées thématiques, temporelles et de réputation pour chaque film.

### A. Données d'Identité & Temporelles (Entrées Catégorielles)

1. **`movieId` (ID Film)**
   - *Source* : `movies.dat` et `ratings.dat`.
   - *Cardinalité* : 3 883 films uniques.
   - *Traitement* : Indexation de $0$ à $M-1$.
   - *Dimension de l'embedding* : 32.

2. **`decade` (Décennie de sortie)**
   - *Source* : Extraite de l'année incluse dans le titre dans `movies.dat` (ex: `"Toy Story (1995)"` $\rightarrow$ `1995` $\rightarrow$ `'1990s'`).
   - *Tranches disponibles* : 10 décennies de `'1910s'` à `'2000s'`.
   - *Traitement* : Mapping vers des index séquentiels de $0$ à $9$.
   - *Dimension de l'embedding* : 4.

---

### B. Caractéristiques Thématiques & Statistiques (Entrées Numériques)

Ces caractéristiques sont regroupées dans un tenseur continu de dimension **20** :

3. **Multi-hot des Genres (`genre_multi_hot` : dim 18)**
   - *Format* : Vecteur binaire où un `1.0` est placé si le film appartient au genre thématique, sinon `0.0`.
   - *Exemple* : Si le film est classé en `Action|Sci-Fi`, il aura `1.0` sur ces deux indices et `0.0` sur les 16 autres.

4. **Statistiques de Réputation (dim 2)**
   - **Popularité (`movie_rating_count`)** : Nombre total de notes reçues par le film sur l'ensemble d'entraînement (normalisé par Z-score).
   - **Appréciation globale (`movie_mean_rating`)** : Note moyenne obtenue par le film sur l'ensemble d'entraînement (normalisée par Z-score).
