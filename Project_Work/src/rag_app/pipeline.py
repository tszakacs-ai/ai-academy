from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from .embedding import AdaEmbeddingModel, LangchainAdaWrapper
from .anonymizer import TextAnonymizer
from .chat_model import ChatCompletionModel
from .bias import BiasChecker

class RAGPipeline:
    def __init__(self) -> None:
        self.anonymizer = TextAnonymizer()
        self.documents = []

        ada_model = AdaEmbeddingModel()
        self.embedding_wrapper = LangchainAdaWrapper(ada_model)
        self.vectorstore = None
        self.chat_model = ChatCompletionModel()
        self.bias_checker = BiasChecker()

        self._build_vectorstore()

    def add_uploaded_files(self, uploaded_files) -> None:
        for uploaded_file in uploaded_files:
            try:
                content = uploaded_file.getvalue().decode("utf-8")
                self.documents.append(
                    Document(page_content=content, metadata={"file_name": uploaded_file.name})
                )
            except Exception as e:  # pragma: no cover - logging
                print(f"Errore nel caricamento file {uploaded_file.name}: {e}")

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
            testo_anonimizzato = self.anonymizer.mask_text(doc.page_content)
            risposta = self.chat_model.ask_about_document(testo_anonimizzato, query)
            if self.bias_checker.is_biased(risposta):
                risposta = "⚠️ Risposta potenzialmente imparziale.\n" + risposta
            risposte += (
                f"**{doc.metadata['file_name']}**\n⚠️ Il contenuto è stato anonimizzato.\n{risposta}\n\n"
            )
        return risposte
