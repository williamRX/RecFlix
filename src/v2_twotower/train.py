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
    Détecte et retourne l'accélérateur matériel disponible (Priorité MPS pour Apple Silicon).
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
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch in dataloader:
        # Transfert de toutes les caractéristiques démographiques et comportementales vers le device
        user_idx = batch["user_idx"].to(device)
        gender_idx = batch["gender_idx"].to(device)
        age_idx = batch["age_idx"].to(device)
        occupation_idx = batch["occupation_idx"].to(device)
        zip_idx = batch["zip_idx"].to(device)
        user_features = batch["user_features"].to(device)
        
        movie_idx = batch["movie_idx"].to(device)
        decade_idx = batch["decade_idx"].to(device)
        movie_features = batch["movie_features"].to(device)
        
        ratings = batch["rating"].to(device)
        
        # Réinitialisation des gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _, _ = model(
            user_idx=user_idx,
            gender_idx=gender_idx,
            age_idx=age_idx,
            occupation_idx=occupation_idx,
            zip_idx=zip_idx,
            user_features=user_features,
            item_idx=movie_idx,
            decade_idx=decade_idx,
            item_features=movie_features
        )
        
        # Calcul de la perte
        loss = criterion(predictions, ratings)
        
        # Rétropropagation
        loss.backward()
        optimizer.step()
        
        # Accumulation
        batch_size = ratings.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    epoch_loss = total_loss / total_samples
    epoch_rmse = epoch_loss ** 0.5
    return epoch_loss, epoch_rmse


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Évalue le modèle sur l'ensemble de validation.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Transfert vers le device
            user_idx = batch["user_idx"].to(device)
            gender_idx = batch["gender_idx"].to(device)
            age_idx = batch["age_idx"].to(device)
            occupation_idx = batch["occupation_idx"].to(device)
            zip_idx = batch["zip_idx"].to(device)
            user_features = batch["user_features"].to(device)
            
            movie_idx = batch["movie_idx"].to(device)
            decade_idx = batch["decade_idx"].to(device)
            movie_features = batch["movie_features"].to(device)
            
            ratings = batch["rating"].to(device)
            
            # Forward pass
            predictions, _, _ = model(
                user_idx=user_idx,
                gender_idx=gender_idx,
                age_idx=age_idx,
                occupation_idx=occupation_idx,
                zip_idx=zip_idx,
                user_features=user_features,
                item_idx=movie_idx,
                decade_idx=decade_idx,
                item_features=movie_features
            )
            
            loss = criterion(predictions, ratings)
            
            batch_size = ratings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
    val_loss = total_loss / total_samples
    val_rmse = val_loss ** 0.5
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
    Orchestre la préparation des données et l'entraînement du modèle Two-Tower.
    """
    device = get_device()
    
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Chargement des données MovieLens 1M
    logging.info("Démarrage du chargement des données...")
    train_loader, val_loader, metadata = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split
    )
    
    # Instanciation
    logging.info("Création du modèle Two-Tower V2...")
    model = TwoTowerModel(
        num_users=metadata["num_users"],
        num_items=metadata["num_movies"],
        num_decades=metadata["num_decades"],
        user_metadata_dim=metadata["user_features_dim"],
        item_metadata_dim=metadata["movie_features_dim"],
        embedding_dim=embedding_dim,
        hidden_dims=[128, 64],
        projection_dim=projection_dim,
        dropout_rate=dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    logging.info(f"Début de l'entraînement pour {epochs} époques...")
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss, train_rmse = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
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
        
        # Enregistrement du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = models_dir / "twotower_best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"✨ Nouveau meilleur modèle sauvegardé (Val MSE: {val_loss:.4f}) dans {best_model_path.name}")
            
    logging.info("🎉 Entraînement terminé !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle Two-Tower V2 (MovieLens 1M).")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques (default: 10)")
    parser.add_argument("--batch_size", type=int, default=512, help="Taille des batchs (default: 512)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Pénalité L2 (default: 1e-4)")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Taille embedding d'IDs (default: 32)")
    parser.add_argument("--projection_dim", type=int, default=32, help="Dimension finale de projection (default: 32)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Taux de dropout (default: 0.2)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Ratio validation (default: 0.2)")
    
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
