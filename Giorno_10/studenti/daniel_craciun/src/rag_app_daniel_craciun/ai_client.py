import os
from dotenv import load_dotenv

class AIProjectClientDefinition:
    load_dotenv()  # Load environment variables from .env file if it exists
    """Generic AI client wrapper. Allows for different providers via dependency injection."""
    def __init__(
        self,
        client_factory=None,
        endpoint_env_var=os.getenv("AZURE_AI_ENDPOINT"),
        api_key_env_var=os.getenv("AZURE_AI_API_KEY"),
    ):
        if not endpoint_env_var:
            raise ValueError(f"{endpoint_env_var} not defined in environment")
        if not api_key_env_var:
            raise ValueError(f"{api_key_env_var} not defined in environment")
        self.endpoint = endpoint_env_var
        self.api_key = api_key_env_var
        if client_factory:
            self.client = client_factory(endpoint_env_var, api_key_env_var)
        else:
            # Default: Azure AI ProjectClient
            from azure.identity import DefaultAzureCredential
            from azure.ai.projects import AIProjectClient
            self.client = AIProjectClient(
                endpoint=self.endpoint,
                azure_endpoint=self.endpoint,
                credential=DefaultAzureCredential(),
                api_key=self.api_key,  # Pass the API key if required by your client
            )
