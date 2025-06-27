import streamlit as st
import os
import glob
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from config import RagClients, AppConfig
from rag_system import search_documents, QueryResult

# --- Dataclasses per la gestione delle email ---
@dataclass
class EmailData:
    """Rappresenta un'email analizzata."""
    filename: str
    content: str
    extracted_questions: List[str]
    customer_type: str
    urgency_level: str
    topics: List[str]

# --- Funzioni di Analisi Email (precedentemente metodi della classe) ---

def extract_questions(content: str) -> List[str]:
    patterns = [r'[.!]?\s*([A-Z][^.!?]*\?)', r'[Pp]otete?\s+([^.!?]*\?)']
    questions = []
    for pattern in patterns:
        questions.extend(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
    return [q.strip() for q in questions if 10 < len(q) < 200][:5]

def determine_customer_type(content: str) -> str:
    content_lower = content.lower()
    if any(w in content_lower for w in ['azienda', 'società', 'ditta']): return "🏭 Azienda"
    if any(w in content_lower for w in ['consulente', 'studio']): return "👨‍💼 Consulente"
    if any(w in content_lower for w in ['laboratorio', 'analisi']): return "🔬 Laboratorio"
    return "👤 Privato"

def determine_urgency(content: str) -> str:
    content_lower = content.lower()
    if any(w in content_lower for w in ['urgente', 'immediato', 'subito']): return "🔴 Alta"
    if any(w in content_lower for w in ['quanto prima', 'appena possibile']): return "🟡 Media"
    return "🟢 Bassa"

def extract_moca_topics(content: str) -> List[str]:
    moca_topics = {
        "Plastica": ["plastica", "polimero", "pvc", "pet"],
        "Certificazioni": ["certificato", "conformità"],
        "Migrazione": ["migrazione", "cessione", "limite"],
        "Test": ["test", "analisi", "prova"],
        "Normative": ["regolamento", "direttiva", "1935/2004"],
    }
    topics = [topic for topic, keywords in moca_topics.items() if any(k in content.lower() for k in keywords)]
    return topics

def analyze_email(filename: str, content: str) -> EmailData:
    return EmailData(
        filename=filename,
        content=content,
        extracted_questions=extract_questions(content),
        customer_type=determine_customer_type(content),
        urgency_level=determine_urgency(content),
        topics=extract_moca_topics(content)
    )

@st.cache_data
def load_emails(config: AppConfig) -> List[EmailData]:
    """Carica e analizza le email dalla cartella specificata."""
    emails = []
    if not os.path.exists(config.EMAIL_FOLDER):
        st.error(f"Cartella email non trovata: {config.EMAIL_FOLDER}")
        return emails
    
    for file_path in glob.glob(os.path.join(config.EMAIL_FOLDER, "*.txt")):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            emails.append(analyze_email(os.path.basename(file_path), content))
        except Exception as e:
            st.warning(f"Errore nel caricare {file_path}: {e}")
    return emails

def generate_email_response(email: EmailData, search_results: List[QueryResult], clients: RagClients, config: AppConfig) -> str:
    """Genera una bozza di risposta email basata su un template specifico."""
    if not clients.chat_client:
        return "Servizio chat non disponibile."
        
    context = "\n\n---\n\n".join([f"Fonte {i+1}: {res.source} (Pag: {res.page_number})\n{res.content}" for i, res in enumerate(search_results)])

    system_prompt = """Sei un consulente MOCA. Genera una risposta email professionale e cortese basandoti ESCLUSIVAMENTE sui documenti normativi forniti.
STILE: Saluto formale, risposta chiara, citazioni normative precise, chiusura cortese, firma "Servizio Consulenza MOCA".
REGOLE: Usa solo le info fornite. Se non hai info sufficienti, dillo chiaramente."""

    user_prompt = f"""EMAIL CLIENTE:
{email.content[:1000]}

DOMANDE ESTRATTE:
- {'- '.join(email.extracted_questions)}

DOCUMENTI NORMATIVI DI RIFERIMENTO:
{context}

Genera una bozza di risposta email per il cliente."""

    try:
        response = clients.chat_client.chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore generazione risposta: {e}"

# --- Funzione principale per la UI della pagina ---
def display_email_page(clients: RagClients, config: AppConfig):
    st.header("📧 Gestione Email Clienti MOCA")
    
    # Carica le email (sfrutta la cache)
    emails = load_emails(config)
    if not emails:
        return

    # Sidebar per filtri e impostazioni
    with st.sidebar:
        st.subheader("Impostazioni RAG Email")
        num_results = st.slider("Documenti da consultare", 1, 10, 5, key="email_rag_slider")
        
        st.subheader("Filtri Email")
        urgency_filter = st.selectbox("Urgenza", ["Tutte", "🔴 Alta", "🟡 Media", "🟢 Bassa"])
        customer_filter = st.selectbox("Tipo Cliente", ["Tutti", "🏭 Azienda", "👨‍💼 Consulente", "🔬 Laboratorio", "👤 Privato"])

    # Filtra email
    filtered_emails = [e for e in emails if (urgency_filter == "Tutte" or e.urgency_level == urgency_filter) and (customer_filter == "Tutti" or e.customer_type == customer_filter)]

    # Sezione principale
    if not filtered_emails:
        st.warning("Nessuna email corrisponde ai filtri selezionati.")
        return
    
    selected_idx = st.selectbox("Seleziona un'email da processare:", range(len(filtered_emails)), format_func=lambda i: f"{filtered_emails[i].filename} ({filtered_emails[i].customer_type})")
    email = filtered_emails[selected_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dettaglio Email")
        st.info(f"**Cliente:** {email.customer_type} | **Urgenza:** {email.urgency_level}")
        st.text_area("Contenuto:", email.content, height=300)
        if email.extracted_questions:
            st.write("**Domande identificate:**")
            for q in email.extracted_questions: st.markdown(f"- *{q}*")

    with col2:
        st.subheader("Assistente Risposta")
        if st.button("🚀 Genera Bozza di Risposta", type="primary"):
            query = email.content + " " + " ".join(email.extracted_questions)
            with st.spinner("Ricerca e generazione in corso..."):
                search_results = search_documents(query, clients, config, top_k=num_results)
                if search_results:
                    response = generate_email_response(email, search_results, clients, config)
                    st.session_state[f'email_response_{email.filename}'] = response
                    st.session_state[f'email_sources_{email.filename}'] = search_results
                else:
                    st.warning("Nessun documento rilevante trovato.")
        
        if f'email_response_{email.filename}' in st.session_state:
            st.text_area("Bozza di risposta:", st.session_state[f'email_response_{email.filename}'], height=300)
            with st.expander("Mostra fonti utilizzate"):
                sources = st.session_state[f'email_sources_{email.filename}']
                for s in sources:
                    st.caption(f"**{s.source} (Pag: {s.page_number}, Score: {s.score:.2f})**: {s.content[:150]}...")
