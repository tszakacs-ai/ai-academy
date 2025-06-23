import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
 
class AzureFoundryOpenAIClient:
    def __init__(self, project_endpoint: str = None, model_name: str = "gpt-4o"):
        """
        Inizializza il client AIProjectClient per Azure AI Projects (Foundry).
        Usa DefaultAzureCredential per autenticazione.
        """
        load_dotenv()
        if project_endpoint is None:
            project_endpoint = os.getenv("AZURE_AIFOUNDRY_ENDPOINT")
        if not project_endpoint:
            raise ValueError("Devi fornire AZURE_AIFOUNDRY_ENDPOINT come parametro o variabile ambiente.")
 
        self.project = AIProjectClient(
            endpoint=project_endpoint,
            credential=DefaultAzureCredential()
        )
        self.model_name = model_name
        self.models = self.project.inference.get_azure_openai_client(api_version="2024-10-21")
 
    def chat_completion(self, messages, max_tokens=256, temperature=0.0, top_p=1.0):
        """
        Esegue la chiamata al modello OpenAI tramite Foundry client
        """
        response = self.models.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
 
    def classifica_documenti(self, cartella_documenti: str = "C:/desktopnodrive/ai-academy/Giorno_6/documenti"):
        """
        Legge file di testo da cartella e li classifica con il modello
        """
        etichette = [
            "Fattura",
            "Contratto",
            "Relazione tecnica",
            "Comunicazione interna",
            "Email commerciale",
            "Altro"
        ]
 
        system_prompt = (
            "Sei un assistente AI incaricato di classificare documenti aziendali.\n"
            "Leggi il contenuto del documento fornito dall'utente e rispondi con UNA SOLA etichetta tra queste:\n"
            f"{', '.join(etichette)}.\n"
            "Non fornire spiegazioni, solo l'etichetta più adatta."
        )
 
        risultati = {}
 
        if not os.path.exists(cartella_documenti):
            raise FileNotFoundError(f"La cartella {cartella_documenti} non esiste.")
 
        for nome_file in os.listdir(cartella_documenti):
            if nome_file.endswith(".txt"):
                percorso = os.path.join(cartella_documenti, nome_file)
                with open(percorso, "r", encoding="utf-8") as f:
                    contenuto = f.read()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": contenuto[:3000]}  # Limite token di sicurezza
                ]
                try:
                    etichetta = self.chat_completion(messages=messages)
                    risultati[nome_file] = etichetta
                    print(f"[{nome_file}] → {etichetta}")
                except Exception as e:
                    print(f"Errore su {nome_file}: {e}")
 
        return risultati
 

if __name__ == "__main__":
    client = AzureFoundryOpenAIClient()
    client.classifica_documenti()
