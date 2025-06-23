from sentence_transformers import SentenceTransformer, util

# Frasi da confrontare
sentences = [
    "Luca ha comprato una macchina nuova.",
    "Luca si è appena comprato una macchina nuova.",
    "Oggi piove molto a Milano."
]

# Carica il modello
model = SentenceTransformer('all-MiniLM-L6-v2')

# Calcola gli embedding
embeddings = model.encode(sentences, convert_to_tensor=True)

# Calcola la similarità coseno
similarity_1_2 = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
similarity_1_3 = util.pytorch_cos_sim(embeddings[0], embeddings[2]).item()

# Mostra i risultati
print(f"Similarità tra frase 1 e 2: {similarity_1_2:.4f}")
print(f"Similarità tra frase 1 e 3: {similarity_1_3:.4f}")
