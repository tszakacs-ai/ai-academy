from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from .loader import DocumentLoader
from .embedding import AdaEmbeddingModel, LangchainAdaWrapper
from .chat_model import ChatCompletionModel


class RAGPipeline:
    def __init__(self, tmp_folder: str) -> None:
        self.loader = DocumentLoader(tmp_folder)
        self.documents: list[Document] = []
        ada_model = AdaEmbeddingModel()
        self.embedding_wrapper = LangchainAdaWrapper(ada_model)
        self.vectorstore = None
        self.retriever = None
        self.chat_model = ChatCompletionModel()

    def add_uploaded_files(self, uploaded_files) -> None:
        docs = self.loader.load_files(uploaded_files)
        self.documents.extend(docs)
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
            return "Nessun documento caricato per la ricerca."

        docs_simili = self.retriever.get_relevant_documents(query)
        if not docs_simili:
            return "Nessun documento rilevante trovato."

        risposte = ""
        for doc in docs_simili:
            risposta = self.chat_model.ask_about_document(doc.page_content, query)
            risposte += f"**{doc.metadata['file_name']}**\n{risposta}\n\n"
        return risposte
