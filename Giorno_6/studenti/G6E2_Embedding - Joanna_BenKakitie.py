import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()   # Carica le variabili nel file .env
 
# Ottieni la chiave e l'enpoint da Azure OpenAI
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("ENDPOINT_URL")

# Ottieni il client
client = AzureOpenAI(
        api_key=subscription_key,
        azure_endpoint=endpoint,
        api_version="2024-12-01-preview"
    )

# Funzione per la vettorizzazione
def embedding(testo):
    response = client.embeddings.create(
        model="text-embedding-ada-002",  # <-- il deployment name su Azure
        input=[testo]
    )
    return response.choices[0].embedding


# Main
if __name__ == '__main__':

    # Frasi da confrontare
    sentences = [
        "Luca ha comprato una macchina nuova.",
        "Luca si è appena comprato una macchina nuova.",
        "Oggi piove molto a Milano."
    ]

    # Lista embedding
    embeddings_list = list()

    for sentence in sentences:
        embeddings_list.append(embedding(sentence))

    if embeddings_list and len(embeddings_list) >= 2:

        # Calcolo embedding
        em1 = np.array(embeddings_list[0]).reshape(1, -1)
        em2 = np.array(embeddings_list[1]).reshape(1, -1)
        em3 = np.array(embeddings_list[2]).reshape(1, -1)

        # Similarità coseno
        similarity_0_1 = cosine_similarity(em1, em2)[0][0]
        similarity_0_2 = cosine_similarity(em1, em3)[0][0]

        # 5. Output
        print("Similarità tra frase 1 e 2:", similarity_0_1)    # simili
        print("Similarità tra frase 1 e 3:", similarity_0_2)    # diversi