import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-05-15",
)

frasi = [
    "Luca si è comprato una macchina nuova.",
    "Luca si è appena comprato un'auto nuova.",
    "Oggi piove molto a Milano.",
]

response = client.embeddings.create(
    model="text-embedding-ada-002",  # oppure il tuo deployment name su Azure
    input=frasi
)

# Estrai gli embedding in ordine
embeddings = [d.embedding for d in response.data]

# Convertili in matrice numpy
embedding_matrix = np.array(embeddings)

# Calcola matrice di similarità
similarity_matrix = cosine_similarity(embedding_matrix)

# Etichette (puoi usare anche solo numeri o abbreviazioni)
etichette = [f"Frase {i+1}" for i in range(len(frasi))]
 
print("\nMatrice di similarità (cosine):\n")
# Stampa intestazione
print("\t" + "\t".join(etichette))
for i, row in enumerate(similarity_matrix):
    valori = "\t".join(f"{val:.3f}" for val in row)
    print(f"{etichette[i]}\t{valori}")

# Confronti espliciti
print("\n Confronti frase a frase:\n")
for i, j in combinations(range(len(frasi)), 2):
    sim = similarity_matrix[i][j]
    print(f"- Similarità tra:")
    print(f"  ➤ \"{frasi[i]}\"")
    print(f"  ➤ \"{frasi[j]}\"")
    print(f"  → Score: {sim:.3f}\n")






