
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
 
load_dotenv()  # Load environment variables from .env file
 
# language_key = os.environ.get('LANGUAGE_KEY')
# language_endpoint = os.environ.get('LANGUAGE_ENDPOINT')
 

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("ENDPOINT_URL")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    api_key=subscription_key,    # <-- La tua chiave API di Azure OpenAI
    azure_endpoint=endpoint,    # <-- Il tuo API endpoint di Azure OpenAI
    api_version="2025-01-01-preview",    # <-- O la versione sul portale Azure
)

response = client.chat.completions.create(
    model="gpt-4.1",  # <-- Il nome esatto del deployment in Azure
    messages=[
        {"role": "system", "content": "Sei un assistente AI."},
        {"role": "user", "content": "Qual è la capitale dell’Italia?"}
    ],
    max_completion_tokens=256,
    temperature=1,
)

print(response.choices[0].message.content)