import argparse
import logging
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.v2_twotower.data_loader import get_dataloaders
from src.v2_twotower.model import TwoTowerModel

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_device() -> torch.device:
    """
    Détecte et retourne l'accélérateur matériel disponible.
    Priorité absolue donnée à Apple Silicon MPS via torch.device("mps").
    
    Returns:
        L'objet torch.device identifié.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Accélération matérielle Apple Silicon activée via torch.device('mps') 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Accélération matérielle CUDA activée. 🖥️")
    else:
        device = torch.device("cpu")
        logging.warning("Aucun accélérateur matériel détecté. Utilisation du CPU. 🖥️")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Entraîne le modèle sur une seule époque.
    
    Args:
        model: Le modèle Two-Tower à entraîner.
        dataloader: Le DataLoader d'entraînement.
        optimizer: L'optimiseur (ex: AdamW).
        criterion: La fonction de perte (ex: MSELoss).
        device: L'appareil cible (MPS, CUDA, CPU).
        
    Returns:
        Un tuple (loss_moyenne, rmse_moyen).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch in dataloader:
        # Transfert des données vers le device cible
        user_idx = batch["user_idx"].to(device)
        movie_idx = batch["movie_idx"].to(device)
        user_features = batch["user_features"].to(device)
        movie_features = batch["movie_features"].to(device)
        ratings = batch["rating"].to(device)
        
        # Réinitialisation des gradients
        optimizer.zero_grad()
        
        # Passage avant (Forward)
        predictions, _, _ = model(user_idx, user_features, movie_idx, movie_features)
        
        # Calcul de la perte
        loss = criterion(predictions, ratings)
        
        # Rétropropagation (Backward)
        loss.backward()
        
        # Mise à jour des poids
        optimizer.step()
        
        # Accumulation des statistiques
        batch_size = ratings.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    epoch_loss = total_loss / total_samples
    epoch_rmse = (epoch_loss) ** 0.5
    return epoch_loss, epoch_rmse


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Évalue le modèle sur l'ensemble de validation.
    
    Args:
        model: Le modèle Two-Tower à évaluer.
        dataloader: Le DataLoader de validation.
        criterion: La fonction de perte.
        device: L'appareil cible (MPS, CUDA, CPU).
        
    Returns:
        Un tuple (loss_validation, rmse_validation).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Transfert des données vers le device
            user_idx = batch["user_idx"].to(device)
            movie_idx = batch["movie_idx"].to(device)
            user_features = batch["user_features"].to(device)
            movie_features = batch["movie_features"].to(device)
            ratings = batch["rating"].to(device)
            
            # Prédiction
            predictions, _, _ = model(user_idx, user_features, movie_idx, movie_features)
            
            # Calcul de la perte
            loss = criterion(predictions, ratings)
            
            # Accumulation
            batch_size = ratings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
    val_loss = total_loss / total_samples
    val_rmse = (val_loss) ** 0.5
    return val_loss, val_rmse


def train_pipeline(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    embedding_dim: int,
    projection_dim: int,
    dropout: float,
    val_split: float
) -> None:
    """
    Pipeline complet de chargement de données, d'initialisation du modèle et d'entraînement.
    
    Args:
        epochs: Nombre d'époques d'entraînement.
        batch_size: Taille des batchs.
        learning_rate: Taux d'apprentissage.
        weight_decay: Terme de régularisation L2.
        embedding_dim: Dimension de l'embedding d'IDs.
        projection_dim: Dimension de la projection Two-Tower.
        dropout: Taux de Dropout.
        val_split: Pourcentage de l'ensemble de validation.
    """
    # Détection du hardware
    device = get_device()
    
    # Résolution des chemins de données
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Chargement des données
    logging.info("Chargement et préparation des données...")
    train_loader, val_loader, metadata = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split
    )
    
    # Instanciation du modèle Two-Tower
    logging.info("Instanciation du modèle Two-Tower...")
    model = TwoTowerModel(
        num_users=metadata["num_users"],
        num_items=metadata["num_movies"],
        user_metadata_dim=metadata["user_features_dim"],
        item_metadata_dim=metadata["movie_features_dim"],
        embedding_dim=embedding_dim,
        hidden_dims=[128, 64],
        projection_dim=projection_dim,
        dropout_rate=dropout
    )
    
    # Envoi du modèle sur l'accélérateur matériel
    model = model.to(device)
    
    # Définition de l'optimiseur et du critère de perte
    # AdamW applique une pénalité L2 propre (weight decay) sur les poids du réseau
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    logging.info(f"Début de l'entraînement pour {epochs} époques...")
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Une époque d'entraînement
        train_loss, train_rmse = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # Une époque de validation
        val_loss, val_rmse = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        epoch_time = time.time() - start_time
        
        logging.info(
            f"Époque {epoch:02d}/{epochs:02d} ({epoch_time:.1f}s) | "
            f"Train MSE: {train_loss:.4f} (RMSE: {train_rmse:.4f}) | "
            f"Val MSE: {val_loss:.4f} (RMSE: {val_rmse:.4f})"
        )
        
        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = models_dir / "twotower_best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"✨ Nouveau meilleur modèle sauvegardé avec un Val MSE de {val_loss:.4f} dans : {best_model_path}")
            
    logging.info("🎉 Fin de l'entraînement du modèle Two-Tower !")


if __name__ == "__main__":
    # Définition et parsing des arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Entraînement du modèle profond Two-Tower de RecFlix.")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques d'entraînement (default: 10)")
    parser.add_argument("--batch_size", type=int, default=256, help="Taille du batch (default: 256)")
    parser.add_argument("--lr", type=float, default=0.001, help="Taux d'apprentissage (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Régularisation L2 / Weight Decay (default: 1e-4)")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension de l'embedding d'IDs utilisateur/film (default: 32)")
    parser.add_argument("--projection_dim", type=int, default=32, help="Dimension finale de projection (default: 32)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Taux de dropout (default: 0.2)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proportion de l'ensemble de validation (default: 0.2)")
    
    args = parser.parse_args()
    
    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        val_split=args.val_split
    )
