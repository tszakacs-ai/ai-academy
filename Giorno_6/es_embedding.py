import os
import numpy as np
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
 
load_dotenv()  # Carica le variabili d'ambiente dal file .env
 
endpoint = os.getenv("PROJECT_ENDPOINT")  
 
project = AIProjectClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential()
)
 
azure_openai = project.inference.get_azure_openai_client(
    api_version="2023-05-15"  # o "2024-10-21" se non usi preview
)  # :contentReference[oaicite:0]{index=0}

# —————————————————————————————
# LE TUE FRASI
# —————————————————————————————
sentences = [
    "Luca ha comprato una macchina nuova.",
    "Luca si è appena comprato una macchina nuova.",
    "Oggi piove molto a Milano."
]

# —————————————————————————————
# RICHIESTA EMBEDDINGS
# —————————————————————————————
response = azure_openai.embeddings.create(
    model="text-embedding-ada-002",  # oppure il tuo deployment name su Azure
    input=sentences
)
# Estrai gli embedding in ordine
embeddings = [d.embedding for d in response.data]
 
# Convertili in matrice numpy
embedding_matrix = np.array(embeddings)
 
# Calcola matrice di similarità
similarity_matrix = cosine_similarity(embedding_matrix)
 
# Etichette (puoi usare anche solo numeri o abbreviazioni)
etichette = [f"Frase {i+1}" for i in range(len(sentences))]
 
print("\nMatrice di similarità (cosine):\n")
# Stampa intestazione
print("\t" + "\t".join(etichette))
for i, row in enumerate(similarity_matrix):
    valori = "\t".join(f"{val:.3f}" for val in row)
    print(f"{etichette[i]}\t{valori}")
 
# Confronti espliciti
print("\n Confronti frase a frase:\n")
for i, j in combinations(range(len(sentences)), 2):
    sim = similarity_matrix[i][j]
    print(f"- Similarità tra:")
    print(f"  ➤ \"{sentences[i]}\"")
    print(f"  ➤ \"{sentences[j]}\"")
    print(f"  → Score: {sim:.3f}\n")