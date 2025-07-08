import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

class AIProjectClientDefinition:
    """Wrapper per il client Azure AI Project."""

    def __init__(self) -> None:
        endpoint = os.getenv("PROJECT_ENDPOINT")
        if not endpoint:
            raise ValueError("PROJECT_ENDPOINT non definito nel .env")
        self.endpoint = endpoint
        self.client = AIProjectClient(
            endpoint=self.endpoint,
            azure_endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
        )
