from fastapi import FastAPI

app = FastAPI(title="RecFlix API", description="API pour la recommandation de films")

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API RecFlix !"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
