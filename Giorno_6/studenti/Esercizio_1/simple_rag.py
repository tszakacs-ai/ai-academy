import os
from openai import AzureOpenAI

class DocumentClassifier:
    """
    Classe per classificare documenti di testo usando Azure AI Foundry con GPT-4o.
    Identifica se un documento è una fattura o un contratto.
    """
    
    def __init__(self, 
                 endpoint: str,
                 api_key: str,
                 api_version: str = "2024-02-15-preview",
                 deployment_name: str = "gpt-4o"):
        """
        Inizializza il classificatore con le credenziali Azure.
        
        Args:
            endpoint: URL dell'endpoint Azure AI Foundry
            api_key: Chiave API per l'autenticazione
            api_version: Versione dell'API Azure OpenAI
            deployment_name: Nome del deployment del modello GPT-4o
        """
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
    
    def classify_document(self, document_text: str):
        """
        Classifica un documento di testo come fattura o contratto.
        
        Args:
            document_text: Il testo del documento da classificare
            
        Returns:
            str: "fattura" o "contratto"
            
        Raises:
            Exception: Se si verifica un errore durante la chiamata API
        """
        try:
            # Prompt per la classificazione
            system_prompt = """Sei un assistente specializzato nella classificazione di documenti.
                            Il tuo compito è analizzare il testo fornito."""

            user_prompt = f"Classifica il seguente documento come fattura, contratto o altro, Non fornire spiegazioni aggiuntive, solo la classificazione:\n\n{document_text}"
            
            # Chiamata all'API
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            # Estrai e pulisci la risposta
            classification = response.choices[0].message.content.strip().lower()
            
            # Validazione della risposta
            if classification in ["fattura", "contratto"]:
                return classification
            else:
                # Fallback se la risposta non è valida
                raise ValueError(f"Risposta non valida dal modello: {classification}")
                
        except Exception as e:
            raise Exception(f"Errore durante la classificazione: {str(e)}")

if __name__ == "__main__":
    classifier = DocumentClassifier(
        endpoint="https://your-resource.openai.azure.com/",
        api_key="your-api-key",
        deployment_name="gpt-4o"
    )

    with open("fattura.txt", "r", encoding="utf-8") as file:
        doc = file.read()

    try:
        risultato = classifier.classify_document(doc.strip())
        print(f"Il documento è classificato come: {risultato}")
    except Exception as e:
        print(f"Errore: {e}")