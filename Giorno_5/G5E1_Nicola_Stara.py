import os
import logging
from typing import List, Dict, Optional, Union
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """Client per interagire con Azure OpenAI Service"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 api_version: Optional[str] = None,
                 use_managed_identity: bool = False):
        """
        Inizializza il client Azure OpenAI
        
        Args:
            api_key: Chiave API di Azure OpenAI (se None, usa variabile d'ambiente)
            azure_endpoint: Endpoint Azure OpenAI (se None, usa variabile d'ambiente)
            api_version: Versione API (se None, usa variabile d'ambiente o default)
            use_managed_identity: Se True, usa Azure Managed Identity invece della API key
        """
        
        # Carica variabili d'ambiente
        load_dotenv()
        
        # Configurazione parametri
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.use_managed_identity = use_managed_identity
        
        # Validazione configurazione
        if not self.azure_endpoint:
            raise ValueError("Azure endpoint è richiesto. Impostare AZURE_OPENAI_ENDPOINT o passarlo come parametro.")
        
        if not self.use_managed_identity and not self.api_key:
            raise ValueError("API key è richiesta quando non si usa Managed Identity. Impostare AZURE_OPENAI_API_KEY.")
        
        # Inizializza client
        self._initialize_client()
        
        logger.info(f"Client Azure OpenAI inizializzato con endpoint: {self.azure_endpoint}")
    
    def _initialize_client(self):
        """Inizializza il client OpenAI"""
        try:
            if self.use_managed_identity:
                # Usa Azure Managed Identity
                credential = DefaultAzureCredential()
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                
                self.client = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token=token.token,
                )
            else:
                # Usa API key
                self.client = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.api_key,
                )
                
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione del client: {e}")
            raise
    
    def chat_completion(self,
                       messages: List[Dict[str, str]],
                       model: str = "gpt-4o",
                       max_tokens: int = 1000,
                       temperature: float = 0.7,
                       top_p: float = 1.0,
                       frequency_penalty: float = 0.0,
                       presence_penalty: float = 0.0,
                       stop: Optional[Union[str, List[str]]] = None,
                       stream: bool = False) -> Union[str, object]:
        """
        Esegue una chat completion con Azure OpenAI
        
        Args:
            messages: Lista di messaggi nel formato [{"role": "system/user/assistant", "content": "testo"}]
            model: Nome del modello deployment su Azure (default: "gpt-4o")
            max_tokens: Numero massimo di token nella risposta
            temperature: Controllo creatività (0.0-2.0)
            top_p: Controllo diversità nucleo (0.0-1.0)
            frequency_penalty: Penalità frequenza (-2.0 a 2.0)
            presence_penalty: Penalità presenza (-2.0 a 2.0)
            stop: Sequenze di stop
            stream: Se True, ritorna un oggetto stream
            
        Returns:
            Stringa con la risposta del modello o oggetto stream se stream=True
        """
        
        try:
            # Validazione parametri
            if not messages:
                raise ValueError("La lista messages non può essere vuota")
            
            if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
                raise ValueError("Ogni messaggio deve avere 'role' e 'content'")
            
            # Chiamata API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Errore nella chiamata chat completion: {e}")
            raise
    
    def simple_chat(self,
                   user_message: str,
                   system_message: str = "Sei un assistente AI utile e preciso.",
                   model: str = "gpt-4o",
                   max_tokens: int = 1000,
                   temperature: float = 0.7) -> str:
        """
        Versione semplificata per chat con un singolo messaggio utente
        
        Args:
            user_message: Messaggio dell'utente
            system_message: Messaggio di sistema (prompt di ruolo)
            model: Nome del modello deployment
            max_tokens: Numero massimo di token
            temperature: Controllo creatività
            
        Returns:
            Risposta del modello come stringa
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def conversation_chat(self,
                         conversation_history: List[Dict[str, str]],
                         new_user_message: str,
                         system_message: str = "Sei un assistente AI utile e preciso.",
                         model: str = "gpt-4o",
                         max_tokens: int = 1000,
                         temperature: float = 0.7) -> tuple:
        """
        Gestisce una conversazione con cronologia
        
        Args:
            conversation_history: Cronologia conversazione (senza system message)
            new_user_message: Nuovo messaggio utente
            system_message: Messaggio di sistema
            model: Nome del modello
            max_tokens: Numero massimo di token
            temperature: Controllo creatività
            
        Returns:
            Tupla (risposta, cronologia_aggiornata)
        """
        
        # Costruisci messaggi completi
        messages = [{"role": "system", "content": system_message}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": new_user_message})
        
        # Ottieni risposta
        response = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Aggiorna cronologia
        updated_history = conversation_history.copy()
        updated_history.append({"role": "user", "content": new_user_message})
        updated_history.append({"role": "assistant", "content": response})
        
        return response, updated_history
    
    def test_connection(self, model: str = "gpt-4o") -> bool:
        """
        Testa la connessione con Azure OpenAI
        
        Args:
            model: Nome del modello da testare
            
        Returns:
            True se la connessione funziona
        """
        
        try:
            response = self.simple_chat(
                user_message="Ciao, questo è un test di connessione. Rispondi semplicemente 'OK'.",
                max_tokens=10,
                temperature=0
            )
            
            logger.info("Test connessione riuscito")
            return True
            
        except Exception as e:
            logger.error(f"Test connessione fallito: {e}")
            return False


# Funzioni di utilità per uso diretto del modulo
def create_client(**kwargs) -> AzureOpenAIClient:
    """Factory function per creare un client"""
    return AzureOpenAIClient(**kwargs)


def quick_chat(user_message: str,
               system_message: str = "Sei un assistente AI utile e preciso.",
               model: str = "gpt-4o",
               max_tokens: int = 1000,
               temperature: float = 0.7,
               **client_kwargs) -> str:
    """
    Funzione di utilità per chat rapide senza creare esplicitamente il client
    
    Args:
        user_message: Messaggio dell'utente
        system_message: Messaggio di sistema
        model: Nome del modello
        max_tokens: Numero massimo di token
        temperature: Controllo creatività
        **client_kwargs: Parametri aggiuntivi per il client
        
    Returns:
        Risposta del modello
    """
    
    client = AzureOpenAIClient(**client_kwargs)
    return client.simple_chat(
        user_message=user_message,
        system_message=system_message,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )


# Esempio di utilizzo
if __name__ == "__main__":
    try:
        # Crea client
        client = AzureOpenAIClient()
        
        # Test connessione
        if client.test_connection():
            print("✅ Connessione Azure OpenAI riuscita!")
            
            # Esempio chat semplice
            response = client.simple_chat(
                user_message="Qual è la capitale dell'Italia?",
                system_message="Sei un esperto di geografia.",
                temperature=0.3
            )
            print(f"Risposta: {response}")
            
            # Esempio conversazione
            history = []
            response1, history = client.conversation_chat(
                conversation_history=history,
                new_user_message="Ciao, come stai?"
            )
            print(f"AI: {response1}")
            
            response2, history = client.conversation_chat(
                conversation_history=history,
                new_user_message="Parlami di Python"
            )
            print(f"AI: {response2}")
            
        else:
            print("❌ Test connessione fallito")
            
    except Exception as e:
        print(f"Errore: {e}")
        print("\nAssicurati di configurare le variabili d'ambiente:")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_VERSION (opzionale)")