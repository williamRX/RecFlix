import sqlite3
from pathlib import Path
from typing import List, Dict, Any

class CommentsDB:
    """
    Gestionnaire de base de données SQLite pour stocker de manière persistante 
    les commentaires des utilisateurs sur les films.
    """
    
    def __init__(self, db_path: Path) -> None:
        """
        Initialise le gestionnaire et s'assure de la présence de la base et de la table.
        """
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Retourne une connexion active avec le row_factory configuré.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """
        Crée la structure des tables (comments, users, user_ratings) si elles n'existent pas.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            # Table des commentaires
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    movieId INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    comment_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table des profils utilisateurs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    gender TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    occupation INTEGER NOT NULL,
                    zip_code TEXT NOT NULL
                )
            """)
            # Table des évaluations de films
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_ratings (
                    username TEXT NOT NULL,
                    movieId INTEGER NOT NULL,
                    rating REAL NOT NULL,
                    PRIMARY KEY (username, movieId),
                    FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def save_user(self, username: str, gender: str, age: int, occupation: int, zip_code: str) -> None:
        """
        Enregistre ou met à jour le profil d'un utilisateur.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (username, gender, age, occupation, zip_code)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    gender=excluded.gender,
                    age=excluded.age,
                    occupation=excluded.occupation,
                    zip_code=excluded.zip_code
                """,
                (username, gender, age, occupation, zip_code)
            )
            conn.commit()

    def get_user(self, username: str) -> Any:
        """
        Récupère les informations de profil d'un utilisateur par son pseudo.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT username, gender, age, occupation, zip_code FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_rating(self, username: str, movie_id: int, rating: float) -> None:
        """
        Enregistre ou met à jour une évaluation (feedback) de film pour un utilisateur donné.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_ratings (username, movieId, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(username, movieId) DO UPDATE SET rating=excluded.rating
                """,
                (username, movie_id, rating)
            )
            conn.commit()

    def get_user_ratings(self, username: str) -> List[Dict[str, Any]]:
        """
        Récupère toutes les évaluations enregistrées pour un utilisateur donné.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT movieId, rating FROM user_ratings WHERE username = ?",
                (username,)
            )
            rows = cursor.fetchall()
            return [{"movieId": int(row["movieId"]), "rating": float(row["rating"])} for row in rows]

    def add_comment(self, movie_id: int, username: str, comment_text: str) -> Dict[str, Any]:
        """
        Enregistre un nouveau commentaire dans la base et le retourne avec son identifiant et sa date.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO comments (movieId, username, comment_text) VALUES (?, ?, ?)",
                (movie_id, username, comment_text)
            )
            conn.commit()
            comment_id = cursor.lastrowid
            
            # Récupère l'élément inséré pour avoir le timestamp généré par SQLite
            cursor.execute(
                "SELECT id, movieId, username, comment_text, created_at FROM comments WHERE id = ?", 
                (comment_id,)
            )
            row = cursor.fetchone()
            return dict(row)

    def get_comments(self, movie_id: int) -> List[Dict[str, Any]]:
        """
        Renvoie la liste des commentaires pour un film donné, du plus récent au plus ancien.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, movieId, username, comment_text, created_at FROM comments WHERE movieId = ? ORDER BY created_at DESC",
                (movie_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
