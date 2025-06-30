import os
from pathlib import Path
from typing import List
import re
 
import streamlit as st
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
 
# --------------------------------------------------
# ⚙️ CONFIGURAZIONE
# --------------------------------------------------
# Variabili sensibili caricate da .env (non definite hardcoded)
load_dotenv()
DEFAULT_FOLDER_PATH = r"C:\\Users\\XA628GA\\OneDrive - EY\\Desktop\\AI Academy\\ai-academy\\Giorno_8\\rag_documents"
 
# Evita il warning di duplicazione libomp (solo se necessario su Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
 
# --------------------------------------------------
# 📥 LOADER DEI FILE DI TESTO
# --------------------------------------------------
class TextFileLoader:
    """Carica tutti i .txt presenti in una cartella e ne restituisce il contenuto."""
 
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Percorso non trovato: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Non è una directory: {folder_path}")
 
    def load(self) -> list[dict]:
        results = []
        for file in self.folder_path.glob("*.txt"):
            try:
                content = file.read_text(encoding="utf-8")
                results.append({"file_name": file.name, "content": content})
            except Exception as e:
                print(f"Errore nella lettura di {file.name}: {e}")
        return results
 
 
# --------------------------------------------------
# 🧠 CLIENT AZURE AI PROJECT
# --------------------------------------------------
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
 
 
# --------------------------------------------------
# 🔎 EMBEDDING MODEL (ADA) WRAPPER
# --------------------------------------------------
class AdaEmbeddingModel(AIProjectClientDefinition):
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        super().__init__()
        self.model_name = model_name
        self.azure_client = self.client.inference.get_azure_openai_client(api_version="2023-05-15")
 
    def embed_text(self, text: str) -> list[float]:
        response = self.azure_client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding
 
 
class LangchainAdaWrapper(Embeddings):
    """Adapter per usare il modello di embedding con LangChain."""
 
    def __init__(self, ada_model: AdaEmbeddingModel):
        self.ada_model = ada_model
 
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.ada_model.embed_text(text) for text in texts]
 
    def embed_query(self, text: str) -> List[float]:
        return self.ada_model.embed_text(text)
 
 
# --------------------------------------------------
# 🛡️ ANONIMIZZATORE DI TESTO
# --------------------------------------------------
class TextAnonymizer:
    """Maschera entità sensibili usando NER + regex."""
 
    def __init__(self):
        model_name = "dslim/bert-base-NER"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
 
        self.regex_patterns = {
            "IBAN": r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b",
            "CF": r"\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b",
            "PHONE": r"\b(\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{5,8}\b",
        }
 
        self.entity_mask_map = {
            "PER": "[NOME]",
            "LOC": "[INDIRIZZO]",
            "ORG": "[AZIENDA]",
            "MISC": "[VARIABILE]",
            "IBAN": "[IBAN]",
            "CF": "[CF]",
            "PHONE": "[TELEFONO]",
        }
 
    def mask_text(self, text: str) -> str:
        masked_text = text
        # Regex masking
        for label, pattern in self.regex_patterns.items():
            matches = re.findall(pattern, masked_text)
            for match in matches:
                masked_text = masked_text.replace(match, self.entity_mask_map.get(label, "[SENSIBILE]"))
        # NER masking
        entities = self.ner_pipeline(masked_text)
        for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
            label = ent["entity_group"]
            if label in self.entity_mask_map:
                masked_text = masked_text[: ent["start"]] + self.entity_mask_map[label] + masked_text[ent["end"] :]
        return masked_text
 
 
# --------------------------------------------------
# 💬 MODELLO CHAT COMPLETION (GPT-4o)
# --------------------------------------------------
class ChatCompletionModel(AIProjectClientDefinition):
    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__()
        self.model_name = model_name
        self.azure_client = self.client.inference.get_azure_openai_client(api_version="2025-01-01-preview")
 
    def ask_about_document(self, content: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Sei un assistente AI specializzato nell'analisi e nella classificazione di documenti testuali.",
            },
            {"role": "user", "content": f"Documento:\n{content}\n\nDomanda: {question}"},
        ]
        response = self.azure_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
        )
        return response.choices[0].message.content
 
# Modifica della RAGPipeline per poter aggiungere documenti dinamicamente
class RAGPipeline:
    def __init__(self, folder_path: str):
        self.anonymizer = TextAnonymizer()
        self.documents = []
        self.folder_path = folder_path
        self.load_documents_from_folder(folder_path)
 
        ada_model = AdaEmbeddingModel()
        self.embedding_wrapper = LangchainAdaWrapper(ada_model)
        self.vectorstore = None
        self.chat_model = ChatCompletionModel()
 
        self._build_vectorstore()
 
    def load_documents_from_folder(self, folder_path):
        loader = TextFileLoader(folder_path)
        results = loader.load()
        for doc in results:
            self.documents.append(Document(page_content=doc["content"], metadata={"file_name": doc["file_name"]}))
 
    def add_uploaded_files(self, uploaded_files):
        for uploaded_file in uploaded_files:
            try:
                content = uploaded_file.getvalue().decode("utf-8")
                self.documents.append(Document(page_content=content, metadata={"file_name": uploaded_file.name}))
            except Exception as e:
                print(f"Errore nel caricamento file {uploaded_file.name}: {e}")
 
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
            testo_anonimizzato = self.anonymizer.mask_text(doc.page_content)
            risposta = self.chat_model.ask_about_document(testo_anonimizzato, query)
            risposte += f"**{doc.metadata['file_name']}**\n⚠️ Il contenuto è stato anonimizzato.\n{risposta}\n\n"
        return risposte
 
# Funzione principale aggiornata per gestire chat multiple e upload file
def main():
    st.set_page_config(page_title="RAG con Azure OpenAI", page_icon="📄", layout="wide")
 
    st.markdown(
        """
<style>
            body {background-color:#f5f7fa;}
            footer {visibility:hidden;}
            header {visibility:hidden;}
            .block-container {padding-top:2rem;}
            .stChatMessage {border-radius:0.75rem;padding:1rem;margin-bottom:0.5rem;}
            .stChatMessage.user {background-color:#c8f6c8;}
            .stChatMessage.assistant {background-color:#ffffff;}
</style>
        """,
        unsafe_allow_html=True,
    )
 
    st.title("📄 RAG con Azure OpenAI")
    st.markdown(
        "Interroga i tuoi documenti locali con embeddings **Ada** e risposte da **GPT-4o**.\n"
        "Crea nuove chat e carica file testuali dalla sidebar."
    )
 
    # Sidebar configurazione
    st.sidebar.header("⚙️ Configurazione")
 
    # Seleziona cartella documenti
    folder_path = st.sidebar.text_input("📂 Cartella documenti", value=DEFAULT_FOLDER_PATH)
 
    # Inizializzazione sessione chat
    if "chats" not in st.session_state:
        st.session_state.chats = []  # Ogni chat è: {"id", "name", "history", "pipeline", "uploaded_file_names"}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
 
    # Pulsante nuova chat
    if st.sidebar.button("➕ Nuova chat"):
        new_chat_id = len(st.session_state.chats)
        try:
            new_pipeline = RAGPipeline(folder_path)
        except Exception as e:
            st.sidebar.error(f"Errore creazione pipeline: {e}")
            return
        st.session_state.chats.append({
            "id": new_chat_id,
            "name": f"Chat #{new_chat_id + 1}",
            "history": [],
            "pipeline": new_pipeline,
            "uploaded_file_names": set(),  # Nuovo: file unici per chat
        })
        st.session_state.active_chat_id = new_chat_id
 
    # Seleziona chat esistente
    if st.session_state.chats:
        chat_names = [chat["name"] for chat in st.session_state.chats]
        selected_chat_name = st.sidebar.selectbox("🗂️ Seleziona chat", chat_names,
                                                  index=st.session_state.active_chat_id or 0)
        st.session_state.active_chat_id = chat_names.index(selected_chat_name)
 
    # Upload file multipli
    uploaded_files = st.sidebar.file_uploader("⬆️ Carica file .txt", type=["txt"], accept_multiple_files=True)
 
    # Se esiste una chat attiva
    if st.session_state.active_chat_id is not None and st.session_state.chats:
        chat = st.session_state.chats[st.session_state.active_chat_id]
        pipeline = chat["pipeline"]
 
        # Gestione caricamento file senza duplicati
        if uploaded_files:
            nuovi_file = []
            for file in uploaded_files:
                if file.name not in chat["uploaded_file_names"]:
                    nuovi_file.append(file)
                    chat["uploaded_file_names"].add(file.name)
 
            if nuovi_file:
                pipeline.add_uploaded_files(nuovi_file)
                st.sidebar.success(f"{len(nuovi_file)} nuovi file aggiunti.")
            else:
                st.sidebar.info("Nessun nuovo file da aggiungere.")
 
        # Mostra i file caricati in chat
        if chat["uploaded_file_names"]:
            with st.expander("📄 File caricati"):
                for name in sorted(chat["uploaded_file_names"]):
                    st.markdown(f"- {name}")
 
        # Mostra cronologia della chat
        for entry in chat["history"]:
            st.chat_message(entry["role"]).markdown(entry["content"])
 
        # Prompt utente
        prompt = st.chat_input("Scrivi la tua domanda…")
        if prompt:
            chat["history"].append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
 
            risposta = pipeline.answer_query(prompt)
            chat["history"].append({"role": "assistant", "content": risposta})
            st.chat_message("assistant").markdown(risposta)
    else:
        st.info("Crea o seleziona una chat dalla sidebar per iniziare.")
 
 
if __name__ == "__main__":
    main()