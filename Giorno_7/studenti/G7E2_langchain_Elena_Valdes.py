from langchain.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_API_KEY_4o"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT_4o"),
    openai_api_version="2024-12-01-preview",
    deployment_name="gpt-4o",
    temperature=0
)

text = "Mario Rossi ha ricevuto un bonifico sull’IBAN IT60X0542811101000000123456."

messages = [
    SystemMessage(content="Sei un assistente che anonimizza i dati sensibili nei testi."),
    HumanMessage(content=f"Trova tutte le entità nel seguente testo (persone, IBAN, ecc.) e restituisci il testo anonimizzato, sostituendo ogni entità con la sua categoria tra parentesi quadre. Testo: \"{text}\". Rispondi solo con il testo anonimizzato.")
]

response = llm(messages)
print("Testo anonimizzato:", response.content)