import openai
from dotenv import load_dotenv
import os

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Leggi le chiavi dal file .env
api_key = os.getenv('OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_ENDPOINT')
api_version = os.getenv('API_VERSION')

testo = ""

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "Sei un analizzatore testi professionista.In base alle mail che trovi nel testo che ti passo, fornisci come prima risposta un riepilogo sulle richieste dei clienti. E poi formula una risposta per ogni cliente sulla base della sua richiesta."},
        {"role": "user", "content": testo}
    ],
    max_completion_tokens=1024,
    temperature=1,
)

print(response.choices[0].message.content)