import streamlit as st
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config import RagClients, AppConfig

@dataclass
class QueryResult:
    """Rappresenta un singolo risultato di una query al database vettoriale."""
    content: str
    score: float
    source: Optional[str] = "N/D"
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

def get_embedding(text: str, clients: RagClients, config: AppConfig) -> Optional[List[float]]:
    """Genera l'embedding per un dato testo usando il client Azure."""
    if not clients.embedding_client:
        return None
    try:
        cleaned_text = text.strip().replace('\n', ' ')
        response = clients.embedding_client.embeddings.create(
            input=cleaned_text,
            model=config.AZURE_EMBEDDING_DEPLOYMENT
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"❌ Errore durante la generazione dell'embedding: {e}")
        return None

def search_documents(query: str, clients: RagClients, config: AppConfig, top_k: int = 5) -> List[QueryResult]:
    """Cerca documenti pertinenti in Pinecone basandosi sulla query."""
    if not clients.pinecone_index or not clients.embedding_client:
        return []
        
    query_embedding = get_embedding(query, clients, config)
    if not query_embedding:
        return []

    try:
        search_results = clients.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        results = []
        for match in search_results.matches:
            metadata = match.metadata or {}
            content = str(metadata.get('text', metadata.get('content', 'Contenuto non disponibile')))
            
            results.append(QueryResult(
                content=content,
                score=match.score,
                source=metadata.get('source', 'Documento MOCA'),
                page_number=metadata.get('page_number', metadata.get('page')),
                metadata=metadata
            ))
        return results
    except Exception as e:
        st.error(f"❌ Errore durante la ricerca su Pinecone: {e}")
        return []

def generate_rag_response(query: str, search_results: List[QueryResult], chat_history: List[Dict], memory_updates: List[str], clients: RagClients, config: AppConfig) -> str:
    """
    Genera una risposta usando un modello di linguaggio, arricchita con i risultati della ricerca,
    la cronologia della chat e gli aggiornamenti della memoria forniti dall'utente.
    """
    if not clients.chat_client:
        return "Servizio di chat non disponibile."

    # Costruisce il contesto dai documenti trovati
    context_from_docs = "\n\n---\n\n".join(
        [f"Fonte: {res.source}, Pagina: {res.page_number}, Rilevanza: {res.score:.2f}\nContenuto: {res.content}" for res in search_results]
    )

    # Costruisce la memoria dinamica dagli aggiornamenti
    memory_context = ""
    if memory_updates:
        updates_str = "\n- ".join(memory_updates)
        memory_context = f"""INFORMAZIONI PRIORITARIE AGGIORNATE DALL'UTENTE (Segui queste istruzioni sopra ogni altra fonte):
- {updates_str}
"""

    system_prompt = f"""
Sei un assistente esperto di normative MOCA (Materiali e Oggetti a Contatto con Alimenti). Il tuo compito è rispondere in modo preciso, basandoti sulle informazioni fornite.

{memory_context}

REGOLE FONDAMENTALI:
1.  **PRIORITÀ MASSIMA**: Le "INFORMAZIONI PRIORITARIE" fornite dall'utente SOSTITUISCONO e ANNULLANO qualsiasi informazione contrastante proveniente dai documenti normativi. Se l'utente ti ha dato un'istruzione (es. "il materiale X è ora ammesso"), devi attenerti a quella.
2.  **BASATI SUI FATTI**: Usa le informazioni dai "DOCUMENTI NORMATIVI" per rispondere. Cita sempre la fonte e la pagina se disponibili.
3.  **GESTISCI L'INCERTEZZA**: Se le informazioni (incluse quelle aggiornate dall'utente) non sono sufficienti per dare una risposta certa, dichiara onestamente "Sulla base delle informazioni a mia disposizione, non è possibile fornire una risposta definitiva su questo punto." Non inventare mai risposte.
4.  **CONSIDERA LA STORIA**: Tieni conto della conversazione precedente per mantenere il contesto.
"""

    # Prepara i messaggi per l'API, includendo la cronologia
    messages = [{"role": "system", "content": system_prompt}]
    
    # Aggiungi la cronologia della chat (ultimi 6 messaggi per brevità)
    messages.extend(chat_history[-6:])

    # Aggiungi la query corrente e il contesto
    user_content = f"""
DOCUMENTI NORMATIVI RECUPERATI:
---
{context_from_docs}
---

DOMANDA DELL'UTENTE:
{query}
"""
    messages.append({"role": "user", "content": user_content})

    try:
        response = clients.chat_client.chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Errore durante la generazione della risposta AI: {e}")
        return "Mi dispiace, si è verificato un errore."
