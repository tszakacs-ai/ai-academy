from sentence_transformers import SentenceTransformer, util

# 1. Caricamento del modello di embedding
# Utilizza il modello pre-addestrato 'all-MiniLM-L6-v2' per calcolare gli embedding delle frasi
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Definizione delle frasi da confrontare
# Frasi di esempio per calcolare la similarità semantica
sentences = [
    "Luca ha comprato una macchina nuova.",
    "Luca si è appena comprato una macchina nuova.",
    "Oggi piove molto a Milano."
]

# 3. Calcolo degli embedding
# Converte le frasi in vettori numerici utilizzando il modello
embeddings = model.encode(sentences, convert_to_tensor=True)

# 4. Calcolo della similarità coseno
# Misura la similarità semantica tra le frasi utilizzando il prodotto scalare normalizzato
similarity_1_2 = util.cos_sim(embeddings[0], embeddings[1]).item()  # Similarità tra frase 1 e 2
similarity_1_3 = util.cos_sim(embeddings[0], embeddings[2]).item()  # Similarità tra frase 1 e 3

# 5. Output dei risultati
# Stampa i valori di similarità calcolati
print("Similarità tra frase 1 e 2:", similarity_1_2)
print("Similarità tra frase 1 e 3:", similarity_1_3)
