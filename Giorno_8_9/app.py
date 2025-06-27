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
