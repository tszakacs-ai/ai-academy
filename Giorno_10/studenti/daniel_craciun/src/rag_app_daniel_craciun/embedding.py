from typing import List
from langchain.embeddings.base import Embeddings
from .ai_client import AIProjectClientDefinition

class GenericEmbeddingModel:
    def __init__(
        self,
        ai_client: AIProjectClientDefinition,
        model_name: str = None,
    ):
        self.client = ai_client.client
        self.model_name = model_name or "text-embedding-ada-002"

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
        )
        return response.data[0].embedding

class LangchainEmbeddingWrapper(Embeddings):
    """Adapter for using a generic embedding model with LangChain."""

    def __init__(self, embedding_model: GenericEmbeddingModel) -> None:
        self.embedding_model = embedding_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embedding_model.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_text(text)
