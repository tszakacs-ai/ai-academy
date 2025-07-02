import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from config import RagClients, AppConfig

@dataclass
class QueryResult:
    """Rappresenta un singolo risultato di una query al database vettoriale."""
    text: str
    score: float
    source: Optional[str] = "N/D"
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

def get_embedding(text: str, clients: RagClients, model_deployment: str) -> Optional[List[float]]:
    """Genera l'embedding per un dato testo usando il client Azure."""
    if not clients.embedding_client:
        return None
    try:
        cleaned_text = text.strip().replace('\n', ' ')
        response = clients.embedding_client.embeddings.create(
            input=cleaned_text,
            model=model_deployment
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Errore durante la generazione dell'embedding: {e}")
        return None

def search_documents(clients: RagClients, query: str, index_name: str, top_k: int = 5) -> List[QueryResult]:
    """
    Cerca documenti rilevanti nel database vettoriale.
    
    Args:
        clients: Oggetto contenente i client necessari
        query: La query di ricerca dell'utente
        index_name: Il nome dell'indice Pinecone da interrogare
        top_k: Numero massimo di risultati da restituire
        
    Returns:
        Lista di risultati rilevanti
    """
    # Se non ci sono client configurati correttamente, ritorna una lista vuota
    if not clients.embedding_client or not clients.pinecone_index:
        st.error("Client di embedding o indice Pinecone non inizializzati.")
        return []
    
    try:
        # Ottieni l'embedding della query
        query_embedding = get_embedding(query, clients, "text-embedding-ada-002")
        if not query_embedding:
            st.error("Impossibile generare l'embedding per la query.")
            return []
        
        # Esegui la ricerca nell'indice vettoriale
        query_results = clients.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Processa i risultati
        results = []
        for match in query_results['matches']:
            # Estrai i metadati
            metadata = match.get('metadata', {})
            source = metadata.get('source', 'Documento non specificato')
            page = metadata.get('page_number')
            
            # Crea l'oggetto risultato
            result = QueryResult(
                text=metadata.get('text', 'Nessun testo disponibile'),
                score=match['score'],
                source=source,
                page_number=page,
                metadata=metadata
            )
            results.append(result)
        
        return results
        
    except Exception as e:
        st.error(f"Errore durante la ricerca dei documenti: {e}")
        return []

def format_context(relevant_docs: List[QueryResult]) -> str:
    """Formatta i documenti rilevanti in un contesto da passare al modello."""
    if not relevant_docs:
        return ""
    
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source_info = f"Fonte: {doc.source}"
        if doc.page_number is not None:
            source_info += f", Pagina: {doc.page_number}"
        
        context_parts.append(f"[DOCUMENTO {i}]\n{doc.text}\n{source_info}\n")
    
    return "\n".join(context_parts)

def extract_sources(relevant_docs: List[QueryResult]) -> List[str]:
    """Estrae e formatta le fonti dai documenti rilevanti."""
    sources = []
    for doc in relevant_docs:
        source = doc.source
        if doc.page_number is not None:
            source += f" (Pagina {doc.page_number})"
        
        # Evita duplicati
        if source not in sources:
            sources.append(source)
    
    return sources

def generate_rag_response(
    clients: RagClients,
    query: str,
    relevant_docs: List[QueryResult],
    model_deployment: str,
    chat_history=None
) -> Tuple[str, List[str]]:
    """
    Genera una risposta basata sui documenti rilevanti trovati.
    
    Args:
        clients: Oggetto contenente i client necessari
        query: La query dell'utente
        relevant_docs: Lista di documenti rilevanti trovati
        model_deployment: Nome del deployment del modello di chat
        chat_history: Storico opzionale della chat
        
    Returns:
        Tupla contenente la risposta generata e la lista delle fonti
    """
    if not clients.chat_client:
        st.error("Client di chat non inizializzato.")
        return "Mi dispiace, non riesco a generare una risposta al momento.", []
    
    try:
        # Prepara il contesto
        context = format_context(relevant_docs)
        
        # Prepara il sistema di prompt
        system_prompt = """
        Sei un consulente esperto in normative sui Materiali e Oggetti a Contatto con Alimenti (MOCA).
        Non sei un fornitore di materiali o prodotti, ma un esperto normativo, quindi accetti solo domande coerenti al tuo ruolo.
        Rispondi alle domande dell'utente in modo accurato, utilizzando solo le informazioni contenute nei documenti di riferimento forniti.
        Se le informazioni non sono sufficienti o non sono presenti nei documenti, ammetti di non avere abbastanza informazioni senza inventare.
        Cita sempre le fonti specifiche quando possibile, inclusi riferimenti a normative, regolamenti o documenti tecnici.
        Mantieni un tono professionale ma accessibile. Riassumi concetti complessi in modo chiaro.
        """
        
        # Prepara il messaggio con la query e il contesto
        if context:
            user_message = f"""
            Domanda: {query}
            
            Documenti di riferimento:
            {context}
            
            Rispondi alla domanda in modo esaustivo basandoti esclusivamente sui documenti forniti. 
            Non inventare informazioni e cita specifici riferimenti normativi quando pertinenti.
            """
        else:
            user_message = f"""
            Domanda: {query}
            
            Non ci sono documenti di riferimento specifici per questa domanda.
            Rispondi con le tue conoscenze generali sul tema MOCA, ma specifica che non disponi di documenti specifici 
            su questo argomento e suggerisci di consultare le normative o le autorità competenti.
            """
        
        # Prepara i messaggi per la chat
        messages = [{"role": "system", "content": system_prompt}]
        
        # Aggiungi lo storico della chat, se presente
        if chat_history:
            for message in chat_history:
                messages.append({"role": message["role"], "content": message["content"]})
        
        # Aggiungi la query corrente
        messages.append({"role": "user", "content": user_message})
        
        # Genera la risposta
        response = clients.chat_client.chat.completions.create(
            model=model_deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Estrai le fonti
        sources = extract_sources(relevant_docs)
        
        return response.choices[0].message.content, sources
        
    except Exception as e:
        st.error(f"Errore durante la generazione della risposta: {e}")
        return f"Mi dispiace, si è verificato un errore durante l'elaborazione della risposta: {e}", []
