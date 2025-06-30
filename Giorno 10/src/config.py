import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone
from dataclasses import dataclass
import urllib3
import certifi

# Carica le variabili d'ambiente dal file .env
load_dotenv()

def find_ssl_cert():
    """
    Cerca il certificato SSL in vari percorsi comuni o in variabili d'ambiente.
    Restituisce il percorso al certificato se trovato, altrimenti None.
    """
    # Check environment variable first (highest priority)
    if cert_path := os.getenv("SSL_CERT_PATH"):
        if os.path.exists(cert_path):
            return cert_path
    
    # Common locations for Zscaler certificates across users
    possible_locations = [
        # User provided path
        os.getenv("SSL_CERT_PATH"),
        # Current user's home directory
        str(Path.home() / "zscaler-ca-bundle.pem"),
        # Specific user path (your current path)
        "C:\\Users\\XM745EF\\zscaler-ca-bundle.pem",
        # Other common locations
        "C:\\zscaler\\zscaler-ca-bundle.pem",
        # Environment specific locations
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../certs/zscaler-ca-bundle.pem"),
        # Standard certifi location as fallback
        certifi.where()
    ]
    
    # Find first existing certificate
    for location in possible_locations:
        if location and os.path.exists(location):
            return location
            
    return None

# Trova il certificato SSL appropriato
CERT_PATH = find_ssl_cert()

def set_ssl_environment():
    """
    Configura variabili d'ambiente per SSL e certificati.
    Utile per ambienti aziendali con proxy come Zscaler.
    
    Returns:
        dict: Information about the SSL configuration
    """
    ssl_info = {
        "cert_path": CERT_PATH,
        "cert_found": bool(CERT_PATH),
        "ssl_configured": False,
        "fallback_used": False
    }
    
    if CERT_PATH and os.path.exists(CERT_PATH):
        # Set environment variables for requests, urllib3, and other libraries
        os.environ['REQUESTS_CA_BUNDLE'] = CERT_PATH
        os.environ['SSL_CERT_FILE'] = CERT_PATH
        os.environ['CURL_CA_BUNDLE'] = CERT_PATH
        ssl_info["ssl_configured"] = True
        st.sidebar.success(f"✅ Certificato SSL trovato: {Path(CERT_PATH).name}")
    else:
        # Fallback to unverified context if needed
        import ssl
        ssl._create_default_https_context = lambda: ssl._create_unverified_context()
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        ssl_info["fallback_used"] = True
        st.sidebar.warning("⚠️ Certificato SSL non trovato. Usando modalità non verificata.")
        st.sidebar.info(
            "Per usare un certificato SSL personalizzato, imposta la variabile d'ambiente "
            "SSL_CERT_PATH con il percorso al file del certificato."
        )
    
    return ssl_info

# Imposta ambiente SSL all'avvio
set_ssl_environment()

@dataclass
class AppConfig:
    """Contiene le configurazioni fisse dell'applicazione."""
    EMAIL_FOLDER: str = "data/emails/anonymized"
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "compliance50")
    AZURE_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    AZURE_CHAT_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    
@dataclass
class RagClients:
    """Contenitore per i client dei servizi AI e DB."""
    chat_client: AzureOpenAI = None
    embedding_client: AzureOpenAI = None
    pinecone_index: Pinecone.Index = None

def setup_clients() -> RagClients:
    """
    Inizializza e restituisce i client per Azure OpenAI e Pinecone.
    Mostra errori in Streamlit se le configurazioni mancano.
    """
    clients = RagClients()
    
    # Configura Azure OpenAI per Chat
    try:
        clients.chat_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )
        # Test di connessione base
        if not clients.chat_client.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY non trovata.")
    except Exception as e:
        st.error(f"❌ Errore configurazione Azure OpenAI (Chat): {e}")

    # Configura Azure OpenAI per Embedding
    try:
        clients.embedding_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
        )
        if not clients.embedding_client.api_key:
            raise ValueError("AZURE_EMBEDDING_API_KEY non trovata.")
    except Exception as e:
        st.error(f"❌ Errore configurazione Azure OpenAI (Embedding): {e}")

    # Configura Pinecone
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY non trovata.")
        
        # Configure Pinecone with the appropriate SSL settings
        pinecone_kwargs = {"api_key": pinecone_api_key}
        
        # Use SSL certificate if available, otherwise disable verification
        if CERT_PATH and os.path.exists(CERT_PATH):
            pinecone_kwargs["ssl_ca_certs"] = CERT_PATH
        else:
            # Fallback to no verification when certificate not available
            pinecone_kwargs["additional_headers"] = {"Verify-SSL": "false"}
        
        pc = Pinecone(**pinecone_kwargs)
        index_name = os.getenv("PINECONE_INDEX_NAME", "compliance50")
        
        if index_name not in pc.list_indexes().names():
             raise ValueError(f"L'indice '{index_name}' non esiste in Pinecone.")
             
        clients.pinecone_index = pc.Index(index_name)
        stats = clients.pinecone_index.describe_index_stats()
        st.sidebar.write(f"Connesso a Pinecone '{index_name}'.")
        st.sidebar.write(f"Documenti indicizzati: {stats.total_vector_count}")

    except Exception as e:
        st.error(f"❌ Errore configurazione Pinecone: {e}")
        
    return clients