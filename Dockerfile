FROM python:3.11-slim

# Configuration de l'environnement
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Création d'un utilisateur non-root (UID 1000 requis par Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Installation des dépendances Python
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copie de l'intégralité du code et des données
COPY --chown=user . $HOME/app

# Exposition du port requis par Hugging Face Spaces
EXPOSE 7860

# Lancement de l'API de recommandation Two-Tower (V2) sur le port 7860
CMD ["uvicorn", "src.v2_twotower.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

