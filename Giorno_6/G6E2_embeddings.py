from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Carica il modello
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Le frasi
sentences = [
    "luca ha comprato una macchina nuova",
    "luca si è appena comprato una macchina nuova",
    "oggi piove molto a milano"
]

# 3. Calcola gli embedding
embeddings = model.encode(sentences)

# 4. Calcola le similarità (cosine similarity)
sim_1_2 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
sim_1_3 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

# 5. Output
print(f"Similarità tra frase 1 e 2 (simili): {sim_1_2:.4f}")
print(f"Similarità tra frase 1 e 3 (diverse): {sim_1_3:.4f}")