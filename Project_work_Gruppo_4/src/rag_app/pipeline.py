import os
import re
from pathlib import Path
from typing import List

import streamlit as st
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .embedding import AdaEmbeddingModel, LangchainAdaWrapper
from .chat_model import ChatCompletionModel

TMP_UPLOADS_PATH = Path("tmp_uploads")
TMP_UPLOADS_PATH.mkdir(exist_ok=True)


class RAGPipeline:
    """Pipeline RAG semplificata."""

    def __init__(self) -> None:
        self.documents: List[Document] = []
        ada_model = AdaEmbeddingModel()
        self.embedding_wrapper = LangchainAdaWrapper(ada_model)
        self.vectorstore = None
        self.retriever = None
        self.chat_model = ChatCompletionModel()

    def add_uploaded_files(self, uploaded_files) -> None:
        new_documents: List[Document] = []
        for file in uploaded_files:
            file_name = file.name
            try:
                if file_name.lower().endswith(".txt"):
                    content = file.getvalue().decode("utf-8")
                    new_documents.append(Document(page_content=content, metadata={"file_name": file_name}))
                elif file_name.lower().endswith(".pdf"):
                    tmp_path = TMP_UPLOADS_PATH / file_name
                    with open(tmp_path, "wb") as f:
                        f.write(file.getvalue())
                    loader = PyPDFLoader(str(tmp_path))
                    pages = loader.load()
                    for page in pages:
                        page.metadata["file_name"] = file_name
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = splitter.split_documents(pages)
                    new_documents.extend(docs)
                    os.remove(tmp_path)
                else:
                    st.warning(f"Tipo di file non supportato: {file_name}. Saltato.")
            except Exception as e:
                st.error(f"Errore durante l'elaborazione del file '{file_name}': {e}")

        if new_documents:
            self.documents.extend(new_documents)
            self._build_vectorstore()
            st.session_state.info_message = f"✅ {len(uploaded_files)} file elaborati e aggiunti."
        else:
            st.session_state.info_message = "ℹ️ Nessun nuovo file valido da elaborare."

    def _build_vectorstore(self) -> None:
        if self.documents:
            try:
                self.vectorstore = FAISS.from_documents(self.documents, embedding=self.embedding_wrapper)
                self.retriever = self.vectorstore.as_retriever()
            except Exception as e:
                st.exception(f"ERRORE: Impossibile creare il Vectorstore. Dettagli: {e}")
                self.vectorstore = None
                self.retriever = None
        else:
            self.vectorstore = None
            self.retriever = None

    def answer_query(self, query: str) -> str:
        if not self.retriever:
            return "Nessun documento caricato o indicizzato per la ricerca. Carica prima dei file."

        try:
            docs_simili = self.retriever.get_relevant_documents(query)
        except Exception as e:
            st.error(f"Errore durante il recupero dei documenti: {e}")
            return f"Spiacente, si è verificato un errore: {e}"

        if not docs_simili:
            return "Nessun documento rilevante trovato per la tua domanda."

        context_content = "\n\n---\n\n".join([doc.page_content for doc in docs_simili])
        link_found = ""
        link_match = re.search(r'(https?://\S+)', context_content)
        if link_match:
            link_found = f"\n\n[Link Rilevante]({link_match.group(0)})"

        risposta = self.chat_model.ask_about_document(context_content, query)
        source_files = ", ".join(list(set([doc.metadata.get("file_name", "N/A") for doc in docs_simili])))
        return f"{risposta}\n\n*Fonte/i: {source_files}*{link_found}"
