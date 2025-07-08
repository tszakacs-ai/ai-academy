import os
import io
from pathlib import Path
from typing import List
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import pandas as pd
import time
import json

# Configurazioni iniziali
load_dotenv()

APP_TITLE = "RAG PDF/TXT con Azure OpenAI"
TMP_UPLOADS_PATH = Path("tmp_uploads")
TMP_UPLOADS_PATH.mkdir(exist_ok=True)

# Classe base Azure AI Project Client
class AIProjectClientDefinition:
    def __init__(self):
        endpoint = os.getenv("PROJECT_ENDPOINT")
        if not endpoint:
            raise ValueError("PROJECT_ENDPOINT non definito nel .env")
        self.endpoint = endpoint
        self.client = AIProjectClient(
            endpoint=self.endpoint,
            azure_endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
        )

# Embedding model Azure OpenAI
class AdaEmbeddingModel(AIProjectClientDefinition):
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        super().__init__()
        self.model_name = model_name
        self.azure_client = self.client.inference.get_azure_openai_client(api_version="2023-05-15")

    def embed_text(self, text: str) -> List[float]:
        response = self.azure_client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

class LangchainAdaWrapper(Embeddings):
    def __init__(self, ada_model: AdaEmbeddingModel):
        self.ada_model = ada_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.ada_model.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.ada_model.embed_text(text)

# Chat Model GPT-4o
class ChatCompletionModel(AIProjectClientDefinition):
    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__()
        self.model_name = model_name
        self.azure_client = self.client.inference.get_azure_openai_client(api_version="2025-01-01-preview")

    def ask_about_document(self, content: str, question: str) -> str:
        messages = [
            {"role": "system", "content": "Sei un assistente AI specializzato in analisi di documenti testuali."},
            {"role": "user", "content": f"Documento:\n{content}\n\nDomanda: {question}"},
        ]
        response = safe_gpt_call(
            self.azure_client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
        )
        return response.choices[0].message.content

# Gestione dei documenti e RAG
class RAGPipeline:
    def __init__(self):
        self.documents = []
        ada_model = AdaEmbeddingModel()
        self.embedding_wrapper = LangchainAdaWrapper(ada_model)
        self.vectorstore = None
        self.retriever = None
        self.chat_model = ChatCompletionModel()

    def add_uploaded_files(self, uploaded_files):
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

    def _build_vectorstore(self):
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

# Funzione per gestire i rate limit Azure

def safe_gpt_call(func, *args, max_retries=5, wait_seconds=60, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:
                st.warning(f"Rate limit Azure superato! Aspetto {wait_seconds} secondi... (Tentativo {attempt+1}/{max_retries})")
                time.sleep(wait_seconds)
            else:
                raise e
    raise Exception("Superato il numero massimo di retry per il rate limit.")

# Template delle domande da porre

TEMPLATE_FIELDS = [
    "Ente erogatore", "Titolo dell'avviso", "Descrizione aggiuntiva", "Beneficiari",
    "Apertura", "Chiusura", "Dotazione finanziaria", "Contributo", "Note",
    "Link", "Key Words", "Aperto (si/no)"
]

TEMPLATE_QUESTIONS = {
    "Ente erogatore": "Scrivi solo il nome esatto dell’ente erogatore di questo bando, scegliendolo dalle prime tre pagine. Se non lo trovi, deduci quello più probabile dal testo. Solo il nome, nessuna spiegazione.",
    "Titolo dell'avviso": "Scrivi solo il titolo ufficiale dell’avviso, così come appare o come puoi dedurlo dalle prime tre pagine. Solo la dicitura, nessuna spiegazione.",
    "Descrizione aggiuntiva": "Scrivi una sola frase molto breve (massimo 25 parole) che riassume l’intero bando. Solo la frase, senza spiegazioni.",
    "Beneficiari": "Scrivi solo i beneficiari principali di questo bando, anche dedotti dal testo. Solo l’elenco, senza spiegazioni.",
    "Apertura": "Scrivi solo la data di apertura (formato GG/MM/AAAA), anche dedotta dal testo se non è esplicitata.",
    "Chiusura": "Scrivi solo la data di chiusura (formato GG/MM/AAAA), anche dedotta dal testo se non è esplicitata.",
    "Dotazione finanziaria": "Qual è la dotazione finanziaria totale del bando? Scrivi solo la cifra o il valore principale della dotazione finanziaria, anche se devi dedurlo dal testo.",
    "Contributo": "Qual è il contributo previsto per i beneficiari? Scrivi solo la cifra o percentuale principale del contributo previsto, anche se la deduci dal testo.",
    "Note": "Scrivi solo una nota rilevante, anche se la deduci dal testo. Solo la nota, senza spiegazioni.",
    "Link": "Scrivi solo il link (URL) principale trovato nel testo, oppure deducilo se presente in altro modo.",
    "Key Words": "Scrivi solo tre parole chiave, anche dedotte dal testo, separate da virgola e senza spiegazioni.",
    "Aperto (si/no)": "Rispondi solo con 'si' o 'no' se il bando è ancora aperto; deduci la risposta dal testo e dalle date. Nessuna spiegazione."
}

def fill_template_for_all_files(pipeline, top_k=3):
    file_to_docs = {}
    for doc in pipeline.documents:
        file = doc.metadata["file_name"]
        file_to_docs.setdefault(file, []).append(doc)

    rows = []
    progress = st.progress(0, text="Estrazione in corso...")
    files = list(file_to_docs.keys())
    total_steps = len(files) * len(TEMPLATE_FIELDS)
    step = 0

    for file in files:
        row = {}
        docs_this_file = file_to_docs[file]
        first_3_chunks = docs_this_file[:3]

        for field in TEMPLATE_FIELDS:
            question = TEMPLATE_QUESTIONS[field]
            retriever = FAISS.from_documents(first_3_chunks if field in ["Ente erogatore", "Titolo dell'avviso"] else docs_this_file, pipeline.embedding_wrapper).as_retriever()
            relevant_docs = retriever.get_relevant_documents(question, k=top_k)
            context = "\n\n".join([d.page_content for d in relevant_docs])

            if field == "Descrizione aggiuntiva":
                question = "Scrivi una sola frase molto breve (massimo 25 parole) che riassume il bando."

            prompt = f"""
Testo selezionato del bando (estratto tramite ricerca semantica):
{context}

Domanda: {question}
Rispondi solo sulla base del testo fornito sopra. Non aggiungere spiegazioni.
"""

            answer = pipeline.chat_model.ask_about_document(context, prompt)
            row[field] = answer.strip()

            st.markdown(f"**[{file}]** - Campo: `{field}`\n- Prompt:\n```{prompt[:500]}...```\n- Risposta GPT:\n```{answer[:300]}...```\n---")
            step += 1
            progress.progress(step / total_steps, text=f"Completato {step}/{total_steps}")

        row["File"] = file
        rows.append(row)

    progress.empty()
    return pd.DataFrame(rows)

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="\ud83d\udcc4")
    st.title(APP_TITLE)

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
        st.session_state.uploaded_file_names = set()

    uploaded_files = st.file_uploader("Carica uno o più file .pdf o .txt", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        nuovi_file = [file for file in uploaded_files if file.name not in st.session_state.uploaded_file_names]
        if nuovi_file:
            st.session_state.pipeline.add_uploaded_files(nuovi_file)
            st.session_state.uploaded_file_names.update(file.name for file in nuovi_file)
            st.success(f"{len(nuovi_file)} nuovi file aggiunti.")
        else:
            st.info("Nessun nuovo file da aggiungere.")

    if st.session_state.uploaded_file_names:
        st.markdown("**File caricati:**")
        for f in sorted(st.session_state.uploaded_file_names):
            st.markdown(f"- {f}")

    if st.session_state.pipeline.documents:
        st.markdown("### Compila il template Excel sui bandi caricati")
        if st.button("Estrai dati e scarica Excel"):
            with st.spinner("Estrazione automatica dei dati dai bandi in corso..."):
                df_output = fill_template_for_all_files(st.session_state.pipeline)
                st.success("Estrazione completata! Scarica il file Excel qui sotto.")
                st.dataframe(df_output)
                output = io.BytesIO()
                df_output.to_excel(output, index=False, engine='openpyxl')
                st.download_button(
                    label="\ud83d\udce5 Scarica Excel compilato",
                    data=output.getvalue(),
                    file_name="bandi_compilati.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    query = st.chat_input("Fai una domanda sui bandi caricati...")
    if query:
        st.chat_message("user").markdown(query)
        risposta = st.session_state.pipeline.answer_query(query)
        st.chat_message("assistant").markdown(risposta)

if __name__ == "__main__":
    main()
