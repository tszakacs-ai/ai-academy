from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Carichiamo il modello MiniLM
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Le 3 frasi da confrontare
frasi = [
    "Luca ha comprato una macchina nuova.",
    "Luca si è appena comprato una macchina nuova.",
    "Oggi piove molto a Milano."
]

# 3. Calcolo degli embedding
embeddings = model.encode(frasi)

# 4. Calcolo similarità coseno tra:
#   - prima e seconda frase (simili)
#   - prima e terza frase (diverse)

sim_1_2 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
sim_1_3 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f"Similarità tra frase 1 e 2 (simili): {sim_1_2:.4f}")
print(f"Similarità tra frase 1 e 3 (diverse): {sim_1_3:.4f}")

# Osservazioni
if sim_1_2 > sim_1_3:
    print("\n✅ Le frasi simili hanno vettori più vicini nello spazio degli embedding.")
else:
    print("\n❌ Le frasi simili NON hanno vettori più vicini, controlla il modello o i dati.")
