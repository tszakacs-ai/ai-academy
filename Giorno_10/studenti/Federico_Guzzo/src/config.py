import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone
from dataclasses import dataclass

# Carica le variabili d'ambiente dal file .env
load_dotenv()

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
        
        pc = Pinecone(api_key=pinecone_api_key)
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
