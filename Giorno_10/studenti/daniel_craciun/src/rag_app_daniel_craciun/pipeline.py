from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from .loader import GenericFileLoader
from .embedding import GenericEmbeddingModel, LangchainEmbeddingWrapper
from .anonymizer import TextAnonymizer
from .chat_model import ChatCompletionModel
from .ai_client import AIProjectClientDefinition

class RAGPipeline:
    def __init__(
        self,
        folder_path: str,
        ai_client: AIProjectClientDefinition,
        anonymizer: TextAnonymizer = None,
        loader: GenericFileLoader = None,
        embedding_model: GenericEmbeddingModel = None,
        chat_model: ChatCompletionModel = None,
    ) -> None:
        self.anonymizer = anonymizer or TextAnonymizer()
        self.documents = []
        self.folder_path = folder_path
        self.loader = loader or GenericFileLoader(folder_path)
        self.load_documents_from_folder()
        self.embedding_model = embedding_model or GenericEmbeddingModel(ai_client)
        self.embedding_wrapper = LangchainEmbeddingWrapper(self.embedding_model)
        self.vectorstore = None
        self.chat_model = chat_model or ChatCompletionModel(ai_client)
        self._build_vectorstore()

    def load_documents_from_folder(self) -> None:
        results = self.loader.load()
        for doc in results:
            self.documents.append(
                Document(page_content=doc["content"], metadata={"file_name": doc["file_name"]})
            )

    def add_uploaded_files(self, uploaded_files) -> None:
        for uploaded_file in uploaded_files:
            try:
                content = uploaded_file.getvalue().decode("utf-8")
                self.documents.append(
                    Document(page_content=content, metadata={"file_name": uploaded_file.name})
                )
            except Exception as e:
                print(f"Error loading file {uploaded_file.name}: {e}")
        self._build_vectorstore()

    def _build_vectorstore(self) -> None:
        if self.documents:
            self.vectorstore = FAISS.from_documents(self.documents, embedding=self.embedding_wrapper)
            self.retriever = self.vectorstore.as_retriever()
        else:
            self.vectorstore = None
            self.retriever = None

    def answer_query(self, query: str) -> str:
        if not self.retriever:
            return "No documents loaded for search."
        docs_simili = self.retriever.get_relevant_documents(query)
        if not docs_simili:
            return "No relevant documents found."
        responses = ""
        for doc in docs_simili:
            responses += f"File: {doc.metadata.get('file_name', 'unknown')}\nContent: {doc.page_content}\n\n"
        return responses
