import openai
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
 
load_dotenv()  # Carica le variabili d'ambiente dal file .env
 
endpoint = os.getenv("PROJECT_ENDPOINT")  
 
project = AIProjectClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential()
)
 
models = project.inference.get_azure_openai_client(api_version="2024-10-21")
 
 
response = models.chat.completions.create(
    model="gpt-4o",  # nome del deployment creato nel tuo progetto Foundry
    messages=[
        {"role": "system", "content": "Sei un assistente preparato e puoi rispondere ad ogni domanda, in particolare sei secializzato in aziende di consulenza e big4."},
        {"role": "user", "content": "Ciao, come ti chiami?"}
    ]
)
 
print(response.choices[0].message.content)