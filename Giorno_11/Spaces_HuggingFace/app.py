import streamlit as st
from config import setup_clients, AppConfig
from page_chat import display_chat_page
from page_emails import display_email_page
import os

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
    # Controllo automatico delle variabili d'ambiente richieste
    required_env_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_EMBEDDING_API_KEY",
        "AZURE_EMBEDDING_ENDPOINT",
        "AZURE_EMBEDDING_API_VERSION",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME"
    ]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        st.error(f"❌ Variabili d'ambiente mancanti: {', '.join(missing_vars)}. Assicurati di averle inserite nei 'Secrets' della Space.")
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
