from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# --- CONFIGURAZIONE AZURE OPENAI ---

AZURE_ENDPOINT = ""
AZURE_KEY = ""
AZURE_DEPLOYMENT = "gpt-4.1-mini"  
API_VERSION = "2024-12-01-preview"

# Istanzia il modello Azure OpenAI tramite Langchain
chat = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT,
    openai_api_base=AZURE_ENDPOINT,
    openai_api_key=AZURE_KEY,
    openai_api_version=API_VERSION,
    temperature=0.7
)

# Storico messaggi: sistema + utente
messages = [
    SystemMessage(content="Sei un assistente intelligente."),
    HumanMessage(content="Ciao, puoi spiegarmi come funziona Azure OpenAI?")
]

# Chiedi la risposta al modello
response = chat(messages)

print(response.content)
