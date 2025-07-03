import os
from dotenv import load_dotenv
import openai

class AIProjectClientDefinition:
    load_dotenv()
    """OpenAI client wrapper for Azure OpenAI endpoints."""
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_AI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not defined in environment")
        if not self.api_key:
            raise ValueError("AZURE_AI_API_KEY not defined in environment")
        openai.api_type = "azure"
        openai.api_base = self.endpoint
        openai.api_key = self.api_key
        openai.api_version = self.api_version
        self.client = openai
