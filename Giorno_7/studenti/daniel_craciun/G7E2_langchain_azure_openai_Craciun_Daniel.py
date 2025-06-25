import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()
azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")

llm = AzureChatOpenAI(
    openai_api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version="2024-12-01-preview",
    deployment_name="o4-mini"  # Sostituisci con il nome del tuo deployment
)

response = llm.invoke("Qual è la capitale dell'Italia?")
print(response)