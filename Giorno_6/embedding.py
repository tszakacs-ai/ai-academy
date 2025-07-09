
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


load_dotenv()

# Frasi da confrontare
frasi = [
    "L'intelligenza artificiale sta cambiando il mondo.",
    "Le reti neurali sono alla base del deep learning.",
    "Oggi è una bella giornata di sole."
]

# Imposta endpoint
endpoint = os.getenv("ADA_ENDPOINT")
if not endpoint:
    raise ValueError("Devi definire ADA_ENDPOINT nel file .env")

# Inizializza client Foundry
project = AIProjectClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential()
)

# Ottieni client per i modelli Azure OpenAI
client = project.inference.get_azure_openai_client(api_version="2023-05-15")

# Chiamata embedding
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=frasi
)

# Estrai e converte embedding in matrice numpy
embeddings = [d.embedding for d in response.data]
embedding_matrix = np.array(embeddings)

# Calcola similarità
similarity_matrix = cosine_similarity(embedding_matrix)

# Etichette per stampa
etichette = [f"Frase {i+1}" for i in range(len(frasi))]

print("\n🔢 Matrice di similarità (coseno):\n")
print("\t" + "\t".join(etichette))
for i, row in enumerate(similarity_matrix):
    valori = "\t".join(f"{val:.3f}" for val in row)
    print(f"{etichette[i]}\t{valori}")

print("\n📊 Confronti frase a frase:\n")
for i, j in combinations(range(len(frasi)), 2):
    sim = similarity_matrix[i][j]
    print(f"- Similarità tra:")
    print(f"  ➤ \"{frasi[i]}\"")
    print(f"  ➤ \"{frasi[j]}\"")
    print(f"  → Score: {sim:.3f}\n")
