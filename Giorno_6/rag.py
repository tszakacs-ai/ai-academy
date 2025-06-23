import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path

 
load_dotenv()  

class TextFileLoader:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Percorso non trovato: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Non è una directory: {folder_path}")

    def load(self) -> list[dict]:
        """
        Legge tutti i file .txt nel percorso specificato.
        Ritorna: lista di dizionari con 'file_name' e 'content'
        """
        results = []
        for file in self.folder_path.glob("*.txt"):
            try:
                content = file.read_text(encoding="utf-8")
                results.append({
                    "file_name": file.name,
                    "content": content
                })
            except Exception as e:
                print(f"Errore nella lettura di {file.name}: {e}")
        return results


class AzureChatClient:
    def __init__(self, model_name: str = "gpt-4o-openai"):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.model_name = model_name

        if not all([self.api_key, self.endpoint]):
            raise ValueError("Variabili d'ambiente non impostate correttamente.")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

    def ask_about_document(self, content: str, question: str) -> str:
        messages = [
                    {
                        "role": "system",
                        "content": (
                            "Sei un assistente AI specializzato nell'analisi e nella classificazione di documenti testuali. "
                        ),
                        "role": "user", "content": f"Documento:\n{content}\n\nDomanda: {question}"
                    }
                ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
        )
        return response.choices[0].message.content




if __name__ == "__main__":

    loader = TextFileLoader(r"C:\DesktopNoOneDrive\ai-academy\Giorno_6\rag_documents")
    documents = loader.load()

    client = AzureChatClient()

    question =              """Il tuo compito è esaminare attentamente ciascun documento e identificare il tipo corretto tra le seguenti categorie: "
                            "mail, nota di credito, ordine di acquisto, contratto.\n\n"
                            "Istruzioni:\n"
                            "- Leggi il contenuto del documento almeno due volte prima di fornire una risposta.\n"
                            "- Utilizza esclusivamente il contenuto del documento per determinare la classificazione.\n"
                            "- Assicurati che la classificazione sia coerente con la struttura, il linguaggio e lo scopo del testo.\n\n"
                            "Importante:\n"
                            "- Non fornire spiegazioni, motivazioni o commenti di alcun tipo.\n"
                            "- Rispondi solo e soltanto con una delle seguenti classi, senza aggiungere nulla:\n"
                            "  mail, nota di credito, ordine di acquisto, contratto."""

    for doc in documents:
        print(f"\n Documento: {doc['file_name']}")
        risposta = client.ask_about_document(doc["content"], question)
        print(f" Risposta:\n{risposta}\n")
