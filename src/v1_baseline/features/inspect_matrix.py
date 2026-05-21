import scipy.sparse as sp
from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "features_matrix.npz"

def inspect_matrix():
    if not MATRIX_PATH.exists():
        print(f"❌ Fichier introuvable : {MATRIX_PATH}")
        return
        
    print(f"Lecture du fichier {MATRIX_PATH.name}...\n")
    
    # Chargement de la matrice creuse
    matrix = sp.load_npz(MATRIX_PATH)
    
    print("="*50)
    print("📊 STATISTIQUES DE LA MATRICE")
    print("="*50)
    print(f"• Format technique : {matrix.format.upper()} (Compressed Sparse Row)")
    print(f"• Dimensions       : {matrix.shape[0]} films (Lignes) x {matrix.shape[1]} features (Colonnes)")
    
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero = matrix.nnz
    sparsity = (1.0 - (non_zero / total_elements)) * 100
    
    print(f"• Éléments stockés : {non_zero} valeurs non-nulles")
    print(f"• Clairsemance     : {sparsity:.2f}% de la matrice est remplie de ZÉROS.")
    
    print("\n" + "="*50)
    print("🔍 EXEMPLE : CE QU'IL Y A DANS LA LIGNE 0 (Le 1er Film)")
    print("="*50)
    
    # On extrait uniquement la première ligne
    row_0 = matrix.getrow(0)
    print(f"Le film 0 possède {row_0.nnz} features actives (mots-clés, genres, réalisateur, etc.).\n")
    
    print("Voici un échantillon de 10 features actives pour ce film (Index Colonne -> Poids/Valeur) :")
    indices = row_0.indices
    data = row_0.data
    
    # Chargement du vocabulaire si disponible
    import json
    vocab_path = PROJECT_ROOT / "data" / "processed" / "feature_names.json"
    vocab = None
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    
    # On affiche les 10 premières valeurs de ce film
    for i in range(min(10, len(indices))):
        col_idx = indices[i]
        val = data[i]
        feature_name = f"'{vocab[col_idx]}'" if vocab else "???"
        print(f" -> Colonne n° {col_idx:<5} ({feature_name}) : {val:.4f}")
        
    if not vocab:
        print("\n💡 Note : Sans les 'vocabulaires' de Scikit-Learn, on ne peut pas savoir à quel mot précis correspond la Colonne 1234, mais l'ordinateur s'en fiche pour calculer des distances mathématiques !")
    else:
        print("\n💡 Grâce à ton idée, nous avons pu associer les mots réels à chaque colonne mathématique !")

if __name__ == "__main__":
    inspect_matrix()
