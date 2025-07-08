from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .embedding import AdaEmbeddingModel, LangchainAdaWrapper
from .chat_model import ChatCompletionModel

TMP_UPLOADS_PATH = Path("tmp_uploads")
TMP_UPLOADS_PATH.mkdir(exist_ok=True)


class RAGPipeline:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        ada_model = AdaEmbeddingModel()
        self.embedding_wrapper = LangchainAdaWrapper(ada_model)
        self.vectorstore = None
        self.retriever = None
        self.chat_model = ChatCompletionModel()

    def add_uploaded_files(self, uploaded_files) -> None:
        for file in uploaded_files:
            if file.name.endswith(".txt"):
                content = file.getvalue().decode("utf-8")
                self.documents.append(Document(page_content=content, metadata={"file_name": file.name}))
            elif file.name.endswith(".pdf"):
                tmp_path = TMP_UPLOADS_PATH / file.name
                with open(tmp_path, "wb") as f:
                    f.write(file.getvalue())
                loader = PyPDFLoader(str(tmp_path))
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(pages)
                for doc in docs:
                    self.documents.append(Document(page_content=doc.page_content, metadata={"file_name": file.name}))
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
