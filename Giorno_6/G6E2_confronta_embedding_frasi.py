from sentence_transformers import SentenceTransformer, util

# 1. Scegli il modello di embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Inserisci le frasi
sentences = [
    "Luca ha comprato una macchina nuova.",
    "Luca si è appena comprato una macchina nuova.",
    "Oggi piove molto a Milano."
]

# 3. Calcola gli embedding
embeddings = model.encode(sentences)

# 4. Calcola la similarità (cosine similarity)
similarity_1_2 = util.cos_sim(embeddings[0], embeddings[1]).item()
similarity_1_3 = util.cos_sim(embeddings[0], embeddings[2]).item()

print(f"Similarità tra frase 1 e 2 (simili): {similarity_1_2:.3f}")
print(f"Similarità tra frase 1 e 3 (diverse): {similarity_1_3:.3f}")

# Osserva e discuti
if similarity_1_2 > similarity_1_3:
    print("Le frasi simili hanno vettori più vicini rispetto a quelle diverse.")
else:
    print("Le frasi simili NON hanno vettori più vicini (controlla il modello o le frasi).")