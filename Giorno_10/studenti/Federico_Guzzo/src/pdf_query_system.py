"""
Sistema completo per interrogare documenti PDF usando Pinecone e Azure OpenAI
Utilizza le credenziali Azure OpenAI e l'indice Pinecone 'compdf' esistente
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from src.rag_system import QueryResult
from src.config import RagClients, AppConfig

# Carica variabili d'ambiente
load_dotenv()

def process_pdf_query(
    query: str,
    clients: RagClients,
    config: AppConfig,
    top_k: int = 5
) -> List[QueryResult]:
    """
    Processa una query contro documenti PDF utilizzando embeddings e ricerca vettoriale.
    
    Args:
        query: La query dell'utente
        clients: I client per Azure OpenAI e Pinecone
        config: Configurazione dell'applicazione
        top_k: Numero di risultati da restituire
        
    Returns:
        Lista di risultati rilevanti dal database vettoriale
    """
    from src.rag_system import search_documents
    
    # Usa la funzione search_documents dal modulo rag_system
    results = search_documents(clients, query, config.PINECONE_INDEX_NAME, top_k)
    
    return results

def load_stored_embeddings(file_path: str) -> Dict[str, Any]:
    """
    Carica gli embeddings e i metadati salvati in precedenza.
    
    Args:
        file_path: Percorso al file PKL contenente gli embeddings
        
    Returns:
        Dizionario con i dati caricati
    """
    import pickle
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Errore nel caricamento degli embeddings: {e}")
        return {}

def save_embeddings_to_file(data: Dict[str, Any], file_path: str) -> bool:
    """
    Salva gli embeddings e i metadati su file.
    
    Args:
        data: Dizionario contenente gli embeddings e i metadati
        file_path: Percorso al file PKL dove salvare i dati
        
    Returns:
        True se il salvataggio è riuscito, False altrimenti
    """
    import pickle
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Errore nel salvataggio degli embeddings: {e}")
        return False

def save_metadata_to_json(metadata: Dict[str, Any], file_path: str) -> bool:
    """
    Salva i metadati in formato JSON.
    
    Args:
        metadata: Dizionario contenente i metadati
        file_path: Percorso al file JSON dove salvare i dati
        
    Returns:
        True se il salvataggio è riuscito, False altrimenti
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Errore nel salvataggio dei metadati: {e}")
        return False
