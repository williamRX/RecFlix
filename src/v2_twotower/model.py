import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class UserTower(nn.Module):
    """
    Tour Utilisateur (User Tower) pour l'encodage des caractéristiques utilisateur.
    
    Cette tour combine l'ID utilisateur, ses caractéristiques démographiques catégorielles 
    (sexe, âge, métier, code postal) sous forme d'embeddings dédiés, et son profil comportemental 
    continu pour les projeter dans un espace latent commun.
    """
    
    def __init__(
        self,
        num_users: int,
        embedding_dim: int,
        metadata_dim: int,       # Dimension des features comportementales (ex: 48)
        hidden_dims: List[int],
        projection_dim: int,
        dropout_rate: float = 0.2
    ) -> None:
        """
        Initialise la tour utilisateur.
        """
        super(UserTower, self).__init__()
        
        # 1. Encodage des identifiants et caractéristiques catégorielles
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=2)
        self.age_embedding = nn.Embedding(num_embeddings=7, embedding_dim=4)
        self.occupation_embedding = nn.Embedding(num_embeddings=21, embedding_dim=8)
        self.zip_embedding = nn.Embedding(num_embeddings=10, embedding_dim=4)
        
        # 2. Entrée totale du MLP
        # Concaténation de : ID (embedding) + Sexe (2) + Âge (4) + Métier (8) + Zip (4) + Stats comportementales (metadata_dim)
        input_dim = embedding_dim + 2 + 4 + 8 + 4 + metadata_dim
        
        # 3. Réseau dense multicouche (MLP)
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

    def forward(
        self,
        user_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        age_idx: torch.Tensor,
        occupation_idx: torch.Tensor,
        zip_idx: torch.Tensor,
        user_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Passage avant de la tour utilisateur.
        """
        # Obtenir les représentations d'embeddings
        u_emb = self.user_embedding(user_idx)
        g_emb = self.gender_embedding(gender_idx)
        a_emb = self.age_embedding(age_idx)
        o_emb = self.occupation_embedding(occupation_idx)
        z_emb = self.zip_embedding(zip_idx)
        
        # Concaténation globale de toutes les caractéristiques
        x = torch.cat([u_emb, g_emb, a_emb, o_emb, z_emb, user_features], dim=1)
        
        # Passage dans le réseau dense
        out: torch.Tensor = self.mlp(x)
        return out


class ItemTower(nn.Module):
    """
    Tour Film (Item Tower) pour l'encodage des caractéristiques des films.
    
    Cette tour combine l'ID film (embedding), la décennie de sortie (embedding) 
    et les caractéristiques thématiques (multi-hot genres) et statistiques pour les projeter
    dans le même espace latent que la tour utilisateur.
    """
    
    def __init__(
        self,
        num_items: int,
        num_decades: int,
        embedding_dim: int,
        metadata_dim: int,       # Dimension du vecteur thématique/statistique (ex: 20)
        hidden_dims: List[int],
        projection_dim: int,
        dropout_rate: float = 0.2
    ) -> None:
        """
        Initialise la tour film.
        """
        super(ItemTower, self).__init__()
        
        # 1. Encodage catégoriel
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        self.decade_embedding = nn.Embedding(num_embeddings=num_decades, embedding_dim=4)
        
        # 2. Entrée totale du MLP
        # Concaténation de : ID (embedding) + Décennie (4) + Genres/Stats (metadata_dim)
        input_dim = embedding_dim + 4 + metadata_dim
        
        # 3. Réseau dense multicouche (MLP)
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

    def forward(
        self,
        item_idx: torch.Tensor,
        decade_idx: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Passage avant de la tour film.
        """
        i_emb = self.item_embedding(item_idx)
        d_emb = self.decade_embedding(decade_idx)
        
        # Concaténation globale
        x = torch.cat([i_emb, d_emb, item_features], dim=1)
        
        # Passage dans le réseau dense
        out: torch.Tensor = self.mlp(x)
        return out


class TwoTowerModel(nn.Module):
    """
    Modèle d'orchestration Two-Tower global.
    
    Il orchestre les appels aux tours utilisateur et film, normalise L2 les représentations
    pour obtenir des vecteurs unitaires sur la sphère latente, et calcule la similarité cosinus.
    Cette similarité est projetée sur l'échelle des notes réelles (de 0.5 à 5.0).
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_decades: int,
        user_metadata_dim: int = 48,
        item_metadata_dim: int = 20,
        embedding_dim: int = 32,
        hidden_dims: List[int] = [128, 64],
        projection_dim: int = 32,
        dropout_rate: float = 0.2,
        min_rating: float = 0.5,
        max_rating: float = 5.0
    ) -> None:
        """
        Initialise l'orchestrateur.
        """
        super(TwoTowerModel, self).__init__()
        
        self.min_rating = min_rating
        self.max_rating = max_rating
        
        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=embedding_dim,
            metadata_dim=user_metadata_dim,
            hidden_dims=hidden_dims,
            projection_dim=projection_dim,
            dropout_rate=dropout_rate
        )
        
        self.item_tower = ItemTower(
            num_items=num_items,
            num_decades=num_decades,
            embedding_dim=embedding_dim,
            metadata_dim=item_metadata_dim,
            hidden_dims=hidden_dims,
            projection_dim=projection_dim,
            dropout_rate=dropout_rate
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        age_idx: torch.Tensor,
        occupation_idx: torch.Tensor,
        zip_idx: torch.Tensor,
        user_features: torch.Tensor,
        item_idx: torch.Tensor,
        decade_idx: torch.Tensor,
        item_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcule les projections et la prédiction de note finale.
        """
        # 1. Projections dans l'espace commun
        user_emb = self.user_tower(user_idx, gender_idx, age_idx, occupation_idx, zip_idx, user_features)
        item_emb = self.item_tower(item_idx, decade_idx, item_features)
        
        # 2. Normalisation L2
        user_emb_norm = F.normalize(user_emb, p=2, dim=1)
        item_emb_norm = F.normalize(item_emb, p=2, dim=1)
        
        # 3. Produit scalaire (similarité cosinus car normalisés)
        cos_sim = torch.sum(user_emb_norm * item_emb_norm, dim=1)
        
        # 4. Ajustement d'échelle pour correspondre à l'intervalle [min_rating, max_rating]
        pred_ratings = self.min_rating + (self.max_rating - self.min_rating) * (cos_sim + 1.0) / 2.0
        
        return pred_ratings, user_emb, item_emb


if __name__ == "__main__":
    # Test unitaire rapide des dimensions
    print("--- Test du modèle TwoTowerModel V2 ---")
    
    n_users = 100
    n_items = 500
    n_decades = 10
    
    model = TwoTowerModel(
        num_users=n_users,
        num_items=n_items,
        num_decades=n_decades,
        user_metadata_dim=48,
        item_metadata_dim=20
    )
    
    b_size = 4
    test_u_idx = torch.randint(0, n_users, (b_size,), dtype=torch.long)
    test_g_idx = torch.randint(0, 2, (b_size,), dtype=torch.long)
    test_a_idx = torch.randint(0, 7, (b_size,), dtype=torch.long)
    test_o_idx = torch.randint(0, 21, (b_size,), dtype=torch.long)
    test_z_idx = torch.randint(0, 10, (b_size,), dtype=torch.long)
    test_u_feat = torch.randn((b_size, 48), dtype=torch.float32)
    
    test_i_idx = torch.randint(0, n_items, (b_size,), dtype=torch.long)
    test_d_idx = torch.randint(0, n_decades, (b_size,), dtype=torch.long)
    test_i_feat = torch.randn((b_size, 20), dtype=torch.float32)
    
    preds, u_rep, i_rep = model(
        test_u_idx, test_g_idx, test_a_idx, test_o_idx, test_z_idx, test_u_feat,
        test_i_idx, test_d_idx, test_i_feat
    )
    
    print(f"Predictions shape: {preds.shape} | valeurs : {preds}")
    print(f"User shape       : {u_rep.shape}")
    print(f"Item shape       : {i_rep.shape}")
    print("✅ Le modèle Two-Tower V2 s'initialise et calcule les dimensions correctement !")
