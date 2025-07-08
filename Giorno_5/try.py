import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
 
load_dotenv()  # Carica le variabili d'ambiente dal file .env
 
api_key = os.getenv("AZURE_OPENAI_API_KEY")  # Assicurati di impostare la variabile d'ambiente AZURE_OPENAI_API_KEY
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Assicurati di impostare la variabile d'ambiente AZURE_OPENAI_ENDPOINT
api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # Assicurati di impostare la variabile d'ambiente AZURE_OPENAI_API_VERSION
 
 
client = AzureOpenAI(
    api_version="2025-01-01-preview",  # Assicurati di usare la versione corretta
    azure_endpoint=azure_endpoint,
    api_key=api_key,
)
 
response = client.chat.completions.create(
    model="gpt-4o_deploy",  
    messages=[
        {"role": "system", "content": "Sei un assistente AI."},
        {"role": "user", "content": "Qual è la capitale dell'Italia?"},
        {"role": "user", "content": "Chi era che ha scritto 'La Divina Commedia'?"}
    ],
    max_tokens=256,
    temperature=1,
    top_p=1.0,
)
 
print(response.choices[0].message.content)