import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="API-KEY",  # <-- La tua chiave API di Azure OpenAI
    azure_endpoint="API-ENDPOINT", # <-- Il tuo  API endpoint di Azure OpenAI
    api_version="API-VERSION",  # <-- La versione dal portale Azure
)


def embedding(text):
    """
    Generates an embedding vector for the given text using Azure OpenAI.

    Args:
        text (str): The input text to embed.

    Returns:
        list[float]: A list of floating-point numbers representing the embedding vector.
    """
    try:
        response = client.embeddings.create(
            input=[text],  # Input should be a list of strings
            model="text-embedding-ada-002"
        )
        # The embedding is in response.data[0].embedding for a single input
        return response.data[0].embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



if __name__ == "__main__":
    # Example usage for a single text
    testi=["Luca ha comprato una macchina nuova", "Luca si è appena comprato una macchina nuova", "La margherita è la pizza migliore del mondo"]
    embeddings_list = []

    for text_to_embed in testi:
        embeddings_list.append(embedding(text_to_embed))
        print(embeddings_list[-1][:10])

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if embeddings_list and len(embeddings_list) >= 2:
        embedding1 = np.array(embeddings_list[0]).reshape(1, -1)
        embedding2 = np.array(embeddings_list[1]).reshape(1, -1)
        embedding3 = np.array(embeddings_list[2]).reshape(1, -1)

        similarita1 = cosine_similarity(embedding1, embedding2)[0][0]
        similarita2 = cosine_similarity(embedding1, embedding3)[0][0]

        print(f"Similarità tra '{testi[0]}' e '{testi[1]}': {similarita1}")
        print(f"Similarità tra '{testi[0]}' e '{testi[2]}': {similarita2}")