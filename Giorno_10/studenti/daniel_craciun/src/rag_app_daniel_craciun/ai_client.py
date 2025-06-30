import os

class AIProjectClientDefinition:
    """Generic AI client wrapper. Allows for different providers via dependency injection."""
    def __init__(self, client_factory=None, endpoint_env_var="PROJECT_ENDPOINT"):
        endpoint = os.getenv(endpoint_env_var)
        if not endpoint:
            raise ValueError(f"{endpoint_env_var} not defined in environment")
        self.endpoint = endpoint
        if client_factory:
            self.client = client_factory(endpoint)
        else:
            # Default: Azure AI ProjectClient
            from azure.identity import DefaultAzureCredential
            from azure.ai.projects import AIProjectClient
            self.client = AIProjectClient(
                endpoint=self.endpoint,
                azure_endpoint=self.endpoint,
                credential=DefaultAzureCredential(),
            )
