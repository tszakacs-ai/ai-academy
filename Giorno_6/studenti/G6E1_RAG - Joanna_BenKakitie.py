import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader

import warnings
warnings.filterwarnings("ignore", message="libmagic is unavailable.*")


# Funzione per inviare la domanda al modello Azure OpenAI
def questioning(subscription_key, endpoint, document_text, question):
    client = AzureOpenAI(
        api_key=subscription_key,
        azure_endpoint=endpoint,
        api_version="2025-01-01-preview"
    )

    response = client.chat.completions.create(
        model="gpt-4.1",  # <-- il deployment name su Azure
        messages=[
            {"role": "system", "content": (
                "Sei un assistente AI specializzato in analisi e nella classificazione di documenti testuali."
                )},
            {"role": "user", "content": f"Documento:\n{document_text}\n\nDomanda:\n{question}"}
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=1.0,
    )

    return response.choices[0].message.content

# Main
if __name__ == '__main__':
    load_dotenv()

    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("ENDPOINT_URL")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ragdocs_path = os.path.join(script_dir, "RAGDocs")

    files_directory = DirectoryLoader(ragdocs_path)


    # Caricamento documenti
    documents = files_directory.load()

    # Prompt della domanda
    question = """
                Il tuo compito è di esaminare attentamente ciascun documento e identificare il tipo corretto tra le seguenti categorie:
                mail, nota di credito, ordine di acquisto, contratto.

                Istruzioni:
                - Leggi il contenuto del documento attentamente.
                - Usa solo le informazioni contenute nel documento per rispondere.
                - Rispondi esclusivamente con una delle seguenti classi, senza aggiungere nulla:
                mail, nota di credito, ordine di acquisto, contratto.
                """
    


    # Esecuzione della classificazione per ciascun documento
    for document in documents:
        file_path = os.path.basename(document.metadata['source'])
        print(f"\n📄 Documento: {file_path}")
        
        risposta = questioning(subscription_key, endpoint, file_path, question)
        print(f"📌 Classificazione: {risposta}\n")

