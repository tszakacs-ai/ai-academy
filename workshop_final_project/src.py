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



load_dotenv()

# CONFIG
APP_TITLE = "RAG PDF/TXT con Azure OpenAI"
TMP_UPLOADS_PATH = Path("tmp_uploads")
TMP_UPLOADS_PATH.mkdir(exist_ok=True)

# Azure Embedding & Chat Model
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





import time

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








# 1. Domande da porre per ciascun campo del template
TEMPLATE_FIELDS = [
    "Ente erogatore", "Titolo dell'avviso", "Descrizione aggiuntiva", "Beneficiari",
    "Apertura", "Chiusura", "Dotazione finanziaria", "Contributo", "Note",
    "Link", "Key Words", "Aperto (si/no)"
]

TEMPLATE_QUESTIONS = {
    "Ente erogatore": "Indica il nome esatto dell’ente erogatore di questo bando. Se non presente, lascia vuoto.",
    "Titolo dell'avviso": "Qual è il titolo ufficiale dell’avviso di questo bando? Se non presente, lascia vuoto.",
    "Descrizione aggiuntiva": "Fornisci una breve descrizione aggiuntiva (2-3 frasi) del bando, se disponibile.",
    "Beneficiari": "Elenca i beneficiari previsti da questo bando. Se non sono indicati chiaramente, lascia vuoto.",
    "Apertura": "Qual è la data di apertura del bando (formato GG/MM/AAAA)? Se non presente, lascia vuoto.",
    "Chiusura": "Qual è la data di chiusura del bando (formato GG/MM/AAAA)? Se non presente, lascia vuoto.",
    "Dotazione finanziaria": "Qual è la dotazione finanziaria totale del bando? Se non presente, lascia vuoto.",
    "Contributo": "Qual è il contributo previsto per i beneficiari? Se non presente, lascia vuoto.",
    "Note": "Aggiungi eventuali note rilevanti (massimo 2 frasi). Se non ci sono, lascia vuoto.",
    "Link": "Indica il link ufficiale a questo bando se presente. Altrimenti lascia vuoto.",
    "Key Words": "Scrivi le parole chiave rilevanti per questo bando, separate da virgola.",
    "Aperto (si/no)": "Il bando è ancora aperto? Rispondi solo 'si' o 'no'. Se la data di chiusura è già passata, rispondi 'no'. Se non è possibile determinare, lascia vuoto."
}






def fill_template_for_all_docs(pipeline):
    # Esegue la batch extraction per tutti i documenti caricati
    rows = []
    progress = st.progress(0, text="Estrazione in corso...")
    for idx, doc in enumerate(pipeline.documents):
        row = {}
        for field in TEMPLATE_FIELDS:
            question = TEMPLATE_QUESTIONS[field]
            answer = pipeline.chat_model.ask_about_document(doc.page_content, question)
            row[field] = answer.strip()
        row["File"] = doc.metadata.get("file_name", "")
        rows.append(row)
        progress.progress((idx+1)/len(pipeline.documents), text=f"Completato {idx+1}/{len(pipeline.documents)}")
    progress.empty()
    df = pd.DataFrame(rows)
    # Ensure that exported Excel cells are always filled
    if not df.empty:
        df[TEMPLATE_FIELDS] = df[TEMPLATE_FIELDS].replace("", "Da compilare")
    return df






import json

# Funzioni per estrarre i testi completi per file e compilare il template Excel
def get_full_texts_per_file(pipeline):
    file_to_chunks = {}
    for doc in pipeline.documents:
        file = doc.metadata["file_name"]
        if file not in file_to_chunks:
            file_to_chunks[file] = []
        file_to_chunks[file].append(doc.page_content)
    # Unisci i chunk per ogni file
    return {file: "\n".join(chunks) for file, chunks in file_to_chunks.items()}




# Funzione per estrarre i campi dal documento usando il modello di chat

def extract_fields_from_doc(chat_model, page_content):
    MAX_INPUT_LENGTH = 12000  # puoi variare se necessario
    if len(page_content) > MAX_INPUT_LENGTH:
        st.warning(f"Documento troppo lungo ({len(page_content)} caratteri). Taglio a {MAX_INPUT_LENGTH}.")
        page_content = page_content[:MAX_INPUT_LENGTH]

    st.info(f"Estrazione da file ({len(page_content)} caratteri): anteprima testo ↓")
    st.code(page_content[:400])

    prompt = """
    Estrai dal seguente bando i seguenti campi:
    - Ente erogatore
    - Titolo dell'avviso
    - Descrizione aggiuntiva
    - Beneficiari
    - Apertura
    - Chiusura
    - Dotazione finanziaria
    - Contributo
    - Note
    - Link
    - Key Words
    - Aperto (si/no)

    Restituisci solo le risposte in formato JSON (senza alcun testo extra), con le chiavi corrispondenti.
    Se non hai informazioni su un campo, lascia la stringa vuota ("").
    Esempio di output:
    {
    "Ente erogatore": "",
    "Titolo dell'avviso": "",
    "Descrizione aggiuntiva": "",
    "Beneficiari": "",
    "Apertura": "",
    "Chiusura": "",
    "Dotazione finanziaria": "",
    "Contributo": "",
    "Note": "",
    "Link": "",
    "Key Words": "",
    "Aperto (si/no)": ""
    }

    Bando:
""" + page_content

    # Usa la chiamata GPT-4o
    response = chat_model.ask_about_document(page_content, prompt)
    # Prova a estrarre JSON
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        result = json.loads(response[json_start:json_end])
    except Exception:
        result = {k: "" for k in TEMPLATE_FIELDS}
        st.error(f"Parsing JSON fallito. Risposta GPT:\n{response[:400]}")
    if any(not result[k] for k in TEMPLATE_FIELDS):
        st.warning(f"Campi vuoti per questo file: {[k for k in TEMPLATE_FIELDS if not result[k]]}")
    return result


# Funzione per compilare il template Excel per tutti i file caricati

def fill_template_for_all_files(pipeline):
    full_texts = get_full_texts_per_file(pipeline)
    rows = []
    progress = st.progress(0, text="Estrazione in corso...")
    files = list(full_texts.keys())
    for idx, file in enumerate(files):
        text = full_texts[file]
        result = extract_fields_from_doc(pipeline.chat_model, text)
        row = {field: result.get(field, "") for field in TEMPLATE_FIELDS}
        row["File"] = file
        rows.append(row)
        progress.progress((idx+1)/len(files), text=f"Completato {idx+1}/{len(files)}")
    progress.empty()
    df = pd.DataFrame(rows)
    # Ensure that exported Excel cells are always filled
    if not df.empty:
        df[TEMPLATE_FIELDS] = df[TEMPLATE_FIELDS].replace("", "Da compilare")
    return df





def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📄")
    st.title(APP_TITLE)
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
        st.session_state.uploaded_file_names = set()

    uploaded_files = st.file_uploader(
        "Carica uno o più file .pdf o .txt",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        nuovi_file = []
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_file_names:
                nuovi_file.append(file)
                st.session_state.uploaded_file_names.add(file.name)
        if nuovi_file:
            st.session_state.pipeline.add_uploaded_files(nuovi_file)
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
                df_output = fill_template_for_all_files(st.session_state.pipeline)  # <--- usa questa funzione!
                st.success("Estrazione completata! Scarica il file Excel qui sotto.")
                st.dataframe(df_output)
                from io import BytesIO
                output = BytesIO()
                df_output.to_excel(output, index=False, engine='openpyxl')
                st.download_button(
                    label="📥 Scarica Excel compilato",
                    data=output.getvalue(),
                    file_name="bandi_compilati.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


    # --- Interfaccia domanda/risposta ---
    query = st.chat_input("Fai una domanda sui bandi caricati...")
    if query:
        st.chat_message("user").markdown(query)
        risposta = st.session_state.pipeline.answer_query(query)
        st.chat_message("assistant").markdown(risposta)

if __name__ == "__main__":
    main()


