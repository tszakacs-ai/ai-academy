import streamlit as st
from config import setup_clients, AppConfig
from page_chat import display_chat_page
from page_emails import display_email_page
import os
import threading
from anonymize_mails import anonymize_documents, watch_and_anonymize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configurazione della pagina Streamlit ---
st.set_page_config(
    page_title="Consulente Normativo MOCA AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Titolo Principale dell'App ---
st.title("🤖 Consulente Normativo MOCA AI")
st.markdown("*Un assistente intelligente per navigare la conformità dei Materiali e Oggetti a Contatto con Alimenti.*")

# --- Caricamento delle configurazioni e dei client ---
# Utilizziamo il caching di Streamlit per inizializzare i client una sola volta.
@st.cache_resource
def load_resources():
    """Carica e restituisce i client per AI e DB, e la configurazione."""
    if not os.path.exists('.env'):
        st.error("❌ File .env non trovato! Assicurati di aver configurato le tue chiavi API.")
        st.stop()
    clients = setup_clients()
    config = AppConfig()
    return clients, config

clients, config = load_resources()

# Controlla se i client sono stati caricati correttamente
if not clients.chat_client or not clients.embedding_client or not clients.pinecone_index:
    st.error("Impossibile inizializzare i servizi AI. Controlla le configurazioni nel file .env e riavvia l'app.")
    st.stop()
else:
    st.sidebar.success("Servizi AI connessi.")

# --- Setup automatic document anonymization ---
@st.cache_resource
def start_anonymization_service():
    """Start the document anonymization service in a background thread"""
    # Base directory of the project
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_dir = os.path.dirname(base_dir)
    
    # Define input and output directories
    documents_dir = os.path.join(project_dir, 'data', 'emails', 'documents')
    anonymized_dir = os.path.join(project_dir, 'data', 'emails', 'anonymized')
    
    # Create directories if they don't exist
    os.makedirs(documents_dir, exist_ok=True)
    os.makedirs(anonymized_dir, exist_ok=True)
    
    # Process existing documents
    logger.info("Processing existing documents...")
    anonymized_files = anonymize_documents(documents_dir, anonymized_dir)
    logger.info(f"Processed {len(anonymized_files)} document(s)")
    
    # Start the watcher thread
    watcher_thread = threading.Thread(
        target=watch_and_anonymize,
        args=(documents_dir, anonymized_dir, 60),  # Check every 60 seconds
        daemon=True
    )
    watcher_thread.start()
    return True

# Start the anonymization service
anonymization_started = start_anonymization_service()
if anonymization_started:
    st.sidebar.success("Servizio di anonimizzazione documenti attivo.")

# --- Navigazione tramite Sidebar ---
st.sidebar.title("Navigazione")
page_selection = st.sidebar.radio(
    "Scegli la modalità:",
    ["💬 Chat Normativa", "📧 Gestione Email"],
    help="Seleziona 'Chat Normativa' per interagire direttamente con la base di conoscenza o 'Gestione Email' per analizzare e rispondere alle email dei clienti."
)

st.sidebar.markdown("---")

# --- Routing delle Pagine ---
if page_selection == "💬 Chat Normativa":
    display_chat_page(clients, config)
elif page_selection == "📧 Gestione Email":
    display_email_page(clients, config)
