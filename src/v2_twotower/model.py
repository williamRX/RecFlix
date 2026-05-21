import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class UserTower(nn.Module):
    """
    Tour Utilisateur (User Tower) pour l'encodage des caractéristiques utilisateur.
    
    Cette classe projette l'ID utilisateur (via une couche d'embedding) et ses
    métadonnées statistiques associées dans un espace latent commun.
    """
    
    def __init__(
        self,
        num_users: int,
        embedding_dim: int,
        metadata_dim: int,
        hidden_dims: List[int],
        projection_dim: int,
        dropout_rate: float = 0.2
    ) -> None:
        """
        Initialise la tour utilisateur.
        
        Args:
            num_users: Nombre total d'utilisateurs uniques dans le dataset.
            embedding_dim: Dimension de l'espace d'embedding pour l'ID utilisateur.
            metadata_dim: Dimension des caractéristiques (métadonnées) de l'utilisateur.
            hidden_dims: Liste des dimensions des couches cachées du réseau dense (MLP).
            projection_dim: Dimension finale de la représentation de sortie.
            dropout_rate: Taux de Dropout appliqué entre les couches denses.
        """
        super(UserTower, self).__init__()
        
        # Couche d'embedding pour l'ID utilisateur
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        
        # Entrée du MLP : concaténation de l'embedding utilisateur et des métadonnées
        input_dim = embedding_dim + metadata_dim
        
        # Construction des couches denses (MLP)
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Couche finale de projection vers l'espace commun
        layers.append(nn.Linear(prev_dim, projection_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Passage avant (Forward pass) de la tour utilisateur.
        
        Args:
            user_idx: Tenseur contenant les index d'utilisateurs (long), forme (batch_size,)
            user_features: Tenseur des métadonnées utilisateurs (float), forme (batch_size, metadata_dim)
            
        Returns:
            Vecteur de représentation de l'utilisateur, forme (batch_size, projection_dim)
        """
        # Encodage de l'ID
        emb = self.user_embedding(user_idx)
        
        # Concaténation de l'embedding avec les métadonnées
        x = torch.cat([emb, user_features], dim=1)
        
        # Passage dans le réseau dense
        out: torch.Tensor = self.mlp(x)
        return out


class ItemTower(nn.Module):
    """
    Tour Item (Item Tower / Movie Tower) pour l'encodage des caractéristiques des films.
    
    Cette classe projette l'ID film (via une couche d'embedding), le multi-hot des genres
    et ses métadonnées statistiques dans le même espace latent que la tour utilisateur.
    """
    
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        metadata_dim: int,
        hidden_dims: List[int],
        projection_dim: int,
        dropout_rate: float = 0.2
    ) -> None:
        """
        Initialise la tour item.
        
        Args:
            num_items: Nombre total de films uniques dans le dataset.
            embedding_dim: Dimension de l'espace d'embedding pour l'ID du film.
            metadata_dim: Dimension des caractéristiques (genres + stats) du film.
            hidden_dims: Liste des dimensions des couches cachées du réseau dense (MLP).
            projection_dim: Dimension finale de la représentation de sortie.
            dropout_rate: Taux de Dropout appliqué entre les couches denses.
        """
        super(ItemTower, self).__init__()
        
        # Couche d'embedding pour l'ID du film
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        
        # Entrée du MLP : concaténation de l'embedding du film et des métadonnées (multi-hot genres + stats)
        input_dim = embedding_dim + metadata_dim
        
        # Construction des couches denses (MLP)
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Couche finale de projection vers l'espace commun
        layers.append(nn.Linear(prev_dim, projection_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_idx: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Passage avant (Forward pass) de la tour item.
        
        Args:
            item_idx: Tenseur contenant les index de films (long), forme (batch_size,)
            item_features: Tenseur des métadonnées du film (float), forme (batch_size, metadata_dim)
            
        Returns:
            Vecteur de représentation du film, forme (batch_size, projection_dim)
        """
        # Encodage de l'ID
        emb = self.item_embedding(item_idx)
        
        # Concaténation de l'embedding avec les métadonnées
        x = torch.cat([emb, item_features], dim=1)
        
        # Passage dans le réseau dense
        out: torch.Tensor = self.mlp(x)
        return out


class TwoTowerModel(nn.Module):
    """
    Modèle d'orchestration Two-Tower.
    
    Il contient la tour utilisateur et la tour item, calcule les projections dans
    l'espace commun et détermine la similarité cosinus entre ces projections.
    Il projette ensuite la similarité cosinus (de -1 à 1) sur l'intervalle de note cible
    (ex: de 0.5 à 5.0).
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_metadata_dim: int,
        item_metadata_dim: int,
        embedding_dim: int = 32,
        hidden_dims: List[int] = [128, 64],
        projection_dim: int = 32,
        dropout_rate: float = 0.2,
        min_rating: float = 0.5,
        max_rating: float = 5.0
    ) -> None:
        """
        Initialise le modèle orchestrateur Two-Tower.
        
        Args:
            num_users: Nombre total d'utilisateurs uniques.
            num_items: Nombre total de films uniques.
            user_metadata_dim: Nombre de métadonnées utilisateur.
            item_metadata_dim: Nombre de métadonnées film.
            embedding_dim: Dimension d'embedding pour les IDs d'utilisateurs et de films.
            hidden_dims: Architecture des couches denses communes aux deux tours.
            projection_dim: Dimension finale commune pour le calcul de similarité.
            dropout_rate: Taux de Dropout.
            min_rating: Note minimale possible dans le dataset (par défaut 0.5).
            max_rating: Note maximale possible dans le dataset (par défaut 5.0).
        """
        super(TwoTowerModel, self).__init__()
        
        self.min_rating = min_rating
        self.max_rating = max_rating
        
        # Initialisation de la tour utilisateur
        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=embedding_dim,
            metadata_dim=user_metadata_dim,
            hidden_dims=hidden_dims,
            projection_dim=projection_dim,
            dropout_rate=dropout_rate
        )
        
        # Initialisation de la tour item
        self.item_tower = ItemTower(
            num_items=num_items,
            embedding_dim=embedding_dim,
            metadata_dim=item_metadata_dim,
            hidden_dims=hidden_dims,
            projection_dim=projection_dim,
            dropout_rate=dropout_rate
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        user_features: torch.Tensor,
        item_idx: torch.Tensor,
        item_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passage avant complet du modèle Two-Tower.
        
        Args:
            user_idx: Index d'utilisateurs (batch_size,)
            user_features: Caractéristiques utilisateurs (batch_size, user_metadata_dim)
            item_idx: Index de films (batch_size,)
            item_features: Caractéristiques films (batch_size, item_metadata_dim)
            
        Returns:
            Un tuple contenant :
            - Le tenseur des prédictions (batch_size,) scalé à la plage de notes.
            - La représentation brute de la tour utilisateur (batch_size, projection_dim).
            - La représentation brute de la tour item (batch_size, projection_dim).
        """
        # 1. Projeter l'utilisateur et le film dans l'espace commun
        user_emb = self.user_tower(user_idx, user_features)
        item_emb = self.item_tower(item_idx, item_features)
        
        # 2. Normaliser les embeddings pour calculer la similarité cosinus
        user_emb_norm = F.normalize(user_emb, p=2, dim=1)
        item_emb_norm = F.normalize(item_emb, p=2, dim=1)
        
        # 3. Calcul du produit scalaire (qui équivaut au cosinus car normalisés à 1.0)
        # Sortie de taille (batch_size,)
        cos_sim = torch.sum(user_emb_norm * item_emb_norm, dim=1)
        
        # 4. Projeter la similarité cosinus (de [-1, 1]) sur l'échelle de note cible (de [min_rating, max_rating])
        # Formule : note_min + (note_max - note_min) * (cos_sim + 1) / 2
        pred_ratings = self.min_rating + (self.max_rating - self.min_rating) * (cos_sim + 1.0) / 2.0
        
        return pred_ratings, user_emb, item_emb


if __name__ == "__main__":
    # Test unitaire rapide du modèle
    print("--- Test du modèle TwoTowerModel ---")
    
    # Dimensions de test
    n_users = 100
    n_items = 500
    u_meta_dim = 2
    i_meta_dim = 21
    
    # Instanciation
    model = TwoTowerModel(
        num_users=n_users,
        num_items=n_items,
        user_metadata_dim=u_meta_dim,
        item_metadata_dim=i_meta_dim
    )
    
    # Création de tenseurs bidons
    b_size = 4
    test_u_idx = torch.randint(0, n_users, (b_size,), dtype=torch.long)
    test_u_feat = torch.randn((b_size, u_meta_dim), dtype=torch.float32)
    test_i_idx = torch.randint(0, n_items, (b_size,), dtype=torch.long)
    test_i_feat = torch.randn((b_size, i_meta_dim), dtype=torch.float32)
    
    # Forward
    preds, u_rep, i_rep = model(test_u_idx, test_u_feat, test_i_idx, test_i_feat)
    
    print(f"Predictions : {preds} (Shape: {preds.shape})")
    print(f"User representations shape: {u_rep.shape}")
    print(f"Item representations shape: {i_rep.shape}")
    print("✅ Le modèle Two-Tower s'initialise et calcule les prédictions correctement !")
