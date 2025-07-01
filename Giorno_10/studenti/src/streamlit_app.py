import os
import streamlit as st
from modules.AzureRag import AzureRAGSystem  # classe rag da file AzureRag.py
from pathlib import Path
import shutil
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# === CONFIGURAZIONE CARTELLA TEMP PER DOCUMENTI ===
TEMP_DIR = Path(tempfile.gettempdir()) / "rag_uploaded_docs"
TEMP_DIR.mkdir(exist_ok=True)

# === CONFIGURAZIONE CREDENZIALI AZURE ===
load_dotenv()
AZURE_ENDPOINT_RAG = os.getenv("AZURE_ENDPOINT_RAG")  
API_KEY_RAG = os.getenv("API_KEY_RAG")
EMBEDDING_DEPLOYMENT_RAG = os.getenv("EMBEDDING_DEPLOYMENT_RAG")  
GPT_DEPLOYMENT_RAG = os.getenv("GPT_DEPLOYMENT_RAG")

# === INIZIALIZZA LO STATO DI SESSIONE ===
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# === CONFIGURAZIONE SIDEBAR ===
st.sidebar.title("📂 Gestione Documenti")

# === UPLOAD DOCUMENTI ===
uploaded_files = st.sidebar.file_uploader(
    "Carica documenti TXT",
    type=["txt"],
    accept_multiple_files=True
)

# Processa i file caricati
for file in uploaded_files:
    file_path = TEMP_DIR / file.name
    if file.name not in [f.name for f in st.session_state.uploaded_files]:
        with open(file_path, "wb") as f:
            f.write(file.read())
        st.session_state.uploaded_files.append(file_path)

# Mostra i file caricati
for file_path in st.session_state.uploaded_files:
    st.sidebar.write(f"📄 {file_path.name}")

# === BOTTONE PER INIZIALIZZARE PIPELINE ===
if st.sidebar.button("🔧 Inizializza RAG"):
    if st.session_state.uploaded_files:
        with st.spinner("🔍 Inizializzazione sistema RAG..."):
            st.session_state.pipeline = AzureRAGSystem(
                azure_endpoint=AZURE_ENDPOINT_RAG,
                api_key=API_KEY_RAG,
                embedding_deployment=EMBEDDING_DEPLOYMENT_RAG,
                gpt_deployment=GPT_DEPLOYMENT_RAG
            )

            # Carica i documenti
            documents = []
            for file_path in st.session_state.uploaded_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append({"title": file_path.name, "content": content})

            st.session_state.pipeline.add_documents(documents)

        st.sidebar.success("✅ Pipeline RAG inizializzata con successo!")
    else:
        st.sidebar.warning("⚠️ Carica prima almeno un file.")

# === PULSANTE PER SVUOTARE FILE E SESSIONE ===
if st.sidebar.button("🗑️ Svuota documenti"):
    st.session_state.uploaded_files = []
    st.session_state.pipeline = None
    shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(exist_ok=True)
    st.sidebar.success("✅ Documenti rimossi.")

# === SEZIONE GESTIONE CONVERSAZIONE ===
st.sidebar.markdown("---")
st.sidebar.title("💬 Gestione Chat")

#riassunto conversazione
if st.session_state.pipeline:
    chat_summary = st.session_state.pipeline.get_chat_summary()
    if chat_summary.get("total_exchanges", 0) > 0:
        st.sidebar.write(f"Scambi: {chat_summary['total_exchanges']}")
        st.sidebar.write(f"Messaggi totali: {chat_summary['total_messages']}")
        
        #argomenti recenti
        if chat_summary.get("recent_topics"):
            st.sidebar.write("🔍 Argomenti recenti:")
            for topic in chat_summary["recent_topics"][:3]:
                st.sidebar.write(f"• {topic}...")


col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Nuova Chat"):
        if st.session_state.pipeline:
            st.session_state.pipeline.clear_chat_history()
            st.sidebar.success("✅ Chat resettata!")
        
with col2:
    if st.button("💾 Salva Chat"):
        if st.session_state.pipeline and st.session_state.pipeline.chat_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            filepath = TEMP_DIR / filename
            st.session_state.pipeline.save_chat_history(str(filepath))
            st.sidebar.success(f"Chat salvata: {filename}")
        else:
            st.sidebar.warning("⚠️ Nessuna conversazione da salvare.")

# === CHAT INTERFACE PRINCIPALE ===
st.title("🤖 RAG Chatbot")

if st.session_state.pipeline is None:
    st.info("📋 **Per iniziare:**\n1. Carica i tuoi documenti TXT nella sidebar\n2. Clicca 'Inizializza RAG'\n3. Inizia a fare domande!")
else:
    st.success("✅ Sistema RAG attivo e pronto!")

# === INPUT UTENTE ===
user_input = st.text_input("✏️ Scrivi una domanda sui documenti caricati:", key="user_input")

if st.button("📤 Invia") or user_input:
    if user_input.strip():
        if st.session_state.pipeline is None:
            st.error("⚠️ Inizializza prima la pipeline RAG dalla sidebar.")
        else:
            with st.spinner("🤔 Sto pensando alla risposta..."):
                # Usa il metodo continue_conversation che mantiene la memoria
                result = st.session_state.pipeline.continue_conversation(
                    user_input,
                    top_k=3,
                    max_tokens=2000
                )

# === VISUALIZZAZIONE CONVERSAZIONE ===
if st.session_state.pipeline and st.session_state.pipeline.chat_history:
    st.markdown("### 💬 Conversazione")
    
    #mostra tutti i messaggi
    for i, message in enumerate(st.session_state.pipeline.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):

                content = message["content"]
                if "[Fonti consultate:" in content:
                    content = content.split("[Fonti consultate:")[0].strip()
                
                st.markdown(content)
                
                if message.get("sources"):
                    with st.expander("📚 Fonti consultate"):
                        for source in message["sources"]:
                            st.write(f"📄 **{source['title']}** (rilevanza: {source['score']:.3f})")

# === PANNELLO INFORMAZIONI ===
if st.session_state.pipeline:
    with st.expander("ℹ️ Informazioni Sistema"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Documenti", len(st.session_state.uploaded_files))
        
        with col2:
            st.metric("💾 Chunk indicizzati", len(st.session_state.pipeline.documents))
            
        with col3:
            chat_summary = st.session_state.pipeline.get_chat_summary()
            st.metric("💬 Scambi conversazione", chat_summary.get("total_exchanges", 0))

# === FOOTER ===
st.markdown("---")
st.markdown("🔧 **Funzionalità disponibili:**")
st.markdown("• 🧠 **Memoria conversazionale**: Il chatbot ricorda la conversazione precedente")
st.markdown("• 🔍 **Ricerca semantica**: Trova informazioni rilevanti nei tuoi documenti")
st.markdown("• 📚 **Citazione fonti**: Mostra da quali documenti provengono le informazioni")
st.markdown("• 💾 **Salvataggio chat**: Salva le conversazioni per riferimenti futuri")
