import openai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("OPENAI_API_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

def instanziate_client():
    client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    return client
