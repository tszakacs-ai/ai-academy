from typing import List
from langchain.embeddings.base import Embeddings

from .ai_client import AIProjectClientDefinition

class AdaEmbeddingModel(AIProjectClientDefinition):
    """Azure OpenAI Ada embedding model"""

    def __init__(self, model_name: str = "text-embedding-ada-002") -> None:
        super().__init__()
        self.model_name = model_name
        self.azure_client = self.client.inference.get_azure_openai_client(
            api_version="2023-05-15"
        )

    def embed_text(self, text: str) -> List[float]:
        response = self.azure_client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding


class LangchainAdaWrapper(Embeddings):
    """Adapter for using the embedding model with LangChain."""

    def __init__(self, ada_model: AdaEmbeddingModel) -> None:
        self.ada_model = ada_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.ada_model.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.ada_model.embed_text(text)
