import streamlit as st
import os
import glob
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from config import RagClients, AppConfig
from rag_system import search_documents, QueryResult, generate_rag_response

# --- Dataclasses per la gestione delle email ---
@dataclass
class EmailData:
    """Rappresenta un'email analizzata."""
    filename: str
    content: str
    extracted_questions: List[str]
    customer_type: str
    urgency_level: str
    response_draft: Optional[str] = None

def extract_questions_from_email(email_text: str, client: RagClients) -> List[str]:
    """Estrae domande o richieste da un'email utilizzando il modello di linguaggio."""
    system_message = """
    Sei un assistente specializzato nell'analisi di email relative a normative sui Materiali e Oggetti a Contatto con Alimenti (MOCA).
    Estrai TUTTE le domande, richieste o punti che richiedono una risposta dall'email fornita.
    Considera sia le domande esplicite (con punto interrogativo) che le richieste implicite (es. "vorrei sapere", "necessito chiarimenti").
    Riporta solo le domande o richieste, una per riga, senza numerazione o altri caratteri. Se non ci sono domande, restituisci una lista vuota.
    """

    try:
        response = client.chat_client.chat.completions.create(
            model=st.session_state.app_config.AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Ecco l'email da analizzare:\n\n{email_text}"}
            ],
            temperature=0.0
        )
        
        # Estrai le domande dalla risposta
        questions_text = response.choices[0].message.content.strip()
        
        # Dividi per righe e filtra righe vuote
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        return questions
    except Exception as e:
        st.error(f"Errore nell'estrazione delle domande: {e}")
        return []

def analyze_email_metadata(email_text: str, client: RagClients) -> Dict[str, Any]:
    """Analizza l'email per estrarre metadati come tipo di cliente e livello di urgenza."""
    system_message = """
    Sei un assistente specializzato nell'analisi di email relative a normative sui Materiali e Oggetti a Contatto con Alimenti (MOCA).
    Analizza l'email e fornisci le seguenti informazioni in formato JSON:
    1. customer_type: classifica il mittente come "produttore", "importatore", "distributore", "utilizzatore", "consulente" o "autorità" in base al contenuto.
    2. urgency_level: valuta l'urgenza della richiesta come "bassa", "media" o "alta" in base al linguaggio e al contesto.
    Rispondi solo con il JSON senza ulteriori spiegazioni.
    """

    try:
        response = client.chat_client.chat.completions.create(
            model=st.session_state.app_config.AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Ecco l'email da analizzare:\n\n{email_text}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        # Estrai il JSON dalla risposta
        import json
        metadata = json.loads(response.choices[0].message.content)
        
        return metadata
    except Exception as e:
        st.error(f"Errore nell'analisi dei metadati: {e}")
        return {"customer_type": "non specificato", "urgency_level": "media"}

def generate_email_response(email_content: str, questions: List[str], relevant_docs: List[QueryResult], client: RagClients) -> str:
    """Genera una risposta all'email basata sulle domande estratte e sui documenti trovati."""
    
    # Prepara il contesto dai documenti rilevanti
    context_text = "\n\n".join([f"Documento {i+1}:\n{doc.text}" for i, doc in enumerate(relevant_docs)])
    
    system_message = """
    Sei un consulente esperto in normative sui Materiali e Oggetti a Contatto con Alimenti (MOCA).
    Stai rispondendo a un'email di un cliente che ha alcune domande o richieste.
    Utilizza le informazioni fornite nei documenti di riferimento per formulare una risposta professionale, accurata e completa.
    
    Segui questi principi nella tua risposta:
    1. Mantieni un tono professionale ma cordiale
    2. Struttura la risposta in modo chiaro, utilizzando paragrafi per ogni punto
    3. Cita specifiche normative o regolamenti quando pertinenti
    4. Se una domanda non può essere risposta con le informazioni disponibili, indicalo chiaramente
    5. Concludi offrendo disponibilità per ulteriori chiarimenti
    
    La risposta deve iniziare con "Gentile Cliente," e terminare con "Cordiali saluti,\nServizio Consulenza Normativa MOCA"
    """
    
    user_message = f"""
    Email ricevuta:
    {email_content}
    
    Domande/richieste identificate:
    {chr(10).join(['- ' + q for q in questions])}
    
    Informazioni dai documenti di riferimento:
    {context_text}
    
    Per favore, genera una risposta email completa e professionale.
    """
    
    try:
        response = client.chat_client.chat.completions.create(
            model=st.session_state.app_config.AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Errore nella generazione della risposta: {e}")
        return "Si è verificato un errore nella generazione della risposta. Si prega di riprovare."

def display_email_page(clients: RagClients, config: AppConfig):
    """Visualizza l'interfaccia della pagina di gestione email."""
    # Inizializzazione dello stato della sessione
    if "app_config" not in st.session_state:
        st.session_state.app_config = config
        
    if "selected_email" not in st.session_state:
        st.session_state.selected_email = None
        
    if "email_cache" not in st.session_state:
        st.session_state.email_cache = {}
        
    if "show_response" not in st.session_state:
        st.session_state.show_response = False
    
    # Intestazione della pagina
    st.header("📧 Gestione Email")
    st.write("Analizza e rispondi alle email dei clienti con l'assistenza dell'intelligenza artificiale.")
    
    # Sidebar con l'elenco delle email
    st.sidebar.header("Email disponibili")
    
    # Carica le email dalla cartella
    email_folder = config.EMAIL_FOLDER
    email_files = glob.glob(os.path.join(email_folder, "*.txt"))
    
    if not email_files:
        st.sidebar.warning(f"Nessuna email trovata nella cartella {email_folder}")
    else:
        # Mostra l'elenco delle email nella sidebar
        email_options = [os.path.basename(file) for file in email_files]
        selected_email_file = st.sidebar.selectbox(
            "Seleziona un'email:",
            options=email_options,
            index=0,
            help="Clicca su un'email per visualizzarne il contenuto e analizzarla."
        )
        
        # Quando viene selezionata un'email, aggiorna il contenuto
        if selected_email_file:
            full_path = os.path.join(email_folder, selected_email_file)
            
            # Verifica se abbiamo già caricato e analizzato questa email
            if full_path in st.session_state.email_cache:
                email_data = st.session_state.email_cache[full_path]
            else:
                # Carica il contenuto dell'email
                try:
                    with open(full_path, 'r', encoding='utf-8') as file:
                        email_content = file.read()
                    
                    # Analisi dell'email
                    with st.sidebar.status("Analisi dell'email in corso..."):
                        # Estrai domande e metadati
                        questions = extract_questions_from_email(email_content, clients)
                        metadata = analyze_email_metadata(email_content, clients)
                    
                    # Crea l'oggetto EmailData e lo salva in cache
                    email_data = EmailData(
                        filename=selected_email_file,
                        content=email_content,
                        extracted_questions=questions,
                        customer_type=metadata.get("customer_type", "non specificato"),
                        urgency_level=metadata.get("urgency_level", "media")
                    )
                    
                    st.session_state.email_cache[full_path] = email_data
                    
                except Exception as e:
                    st.error(f"Errore nel caricamento dell'email: {e}")
                    email_data = None
            
            # Salva l'email selezionata nello stato della sessione
            st.session_state.selected_email = email_data
            st.session_state.show_response = False  # Reset view when changing email
    
    # Contenuto principale
    if st.session_state.selected_email:
        email_data = st.session_state.selected_email
        
        # Layout a colonne
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualizza il contenuto dell'email
            st.subheader(f"Email: {email_data.filename}")
            st.text_area("Contenuto:", value=email_data.content, height=300, disabled=True)
        
        with col2:
            # Visualizza i metadati e le domande estratte
            st.subheader("Analisi dell'email")
            
            st.write("**Tipo di cliente:**")
            customer_type = email_data.customer_type.capitalize()
            st.info(customer_type)
            
            st.write("**Livello di urgenza:**")
            urgency_map = {"bassa": "🟢 Bassa", "media": "🟠 Media", "alta": "🔴 Alta"}
            st.warning(urgency_map.get(email_data.urgency_level.lower(), email_data.urgency_level))
            
            st.write("**Domande/richieste identificate:**")
            if email_data.extracted_questions:
                for i, question in enumerate(email_data.extracted_questions, 1):
                    st.markdown(f"{i}. {question}")
            else:
                st.info("Nessuna domanda o richiesta specifica identificata.")
        
        # Bottone per generare risposta
        if st.button("🤖 Genera risposta automatica", key="generate_button"):
            st.session_state.show_response = True
            
            # Ricerca documenti rilevanti per tutte le domande
            with st.status("🔍 Ricerca informazioni rilevanti..."):
                all_relevant_docs = []
                
                # Se ci sono domande specifiche, cerca documenti per ciascuna
                if email_data.extracted_questions:
                    for question in email_data.extracted_questions:
                        docs = search_documents(clients, question, config.PINECONE_INDEX_NAME, top_k=3)
                        all_relevant_docs.extend(docs)
                else:
                    # Altrimenti usa l'intera email come query
                    condensed_content = re.sub(r'\s+', ' ', email_data.content[:500])
                    docs = search_documents(clients, condensed_content, config.PINECONE_INDEX_NAME, top_k=5)
                    all_relevant_docs.extend(docs)
                
                # Rimuovi documenti duplicati basandoti sul testo
                unique_docs = []
                seen_texts = set()
                for doc in all_relevant_docs:
                    if doc.text not in seen_texts:
                        seen_texts.add(doc.text)
                        unique_docs.append(doc)
                
                st.write(f"Trovati {len(unique_docs)} documenti rilevanti.")
            
            # Genera la risposta
            with st.status("🤖 Generazione risposta in corso..."):
                response = generate_email_response(
                    email_data.content, 
                    email_data.extracted_questions, 
                    unique_docs, 
                    clients
                )
                
                # Salva la risposta nell'oggetto email
                email_data.response_draft = response
                st.session_state.email_cache[os.path.join(email_folder, email_data.filename)] = email_data
            
        # Mostra la risposta generata se disponibile e l'utente ha cliccato il bottone
        if st.session_state.show_response and email_data.response_draft:
            st.subheader("✏️ Bozza di risposta")
            
            # Permetti la modifica della risposta
            edited_response = st.text_area(
                "Modifica la risposta se necessario:",
                value=email_data.response_draft,
                height=400
            )
            
            # Opzioni per la risposta
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copia negli appunti", key="copy_button"):
                    st.toast("Risposta copiata negli appunti!")
                    # Nota: il meccanismo di copia effettiva è gestito da JavaScript
                    js_safe_text = edited_response.replace('"', '\\"')
                    st.markdown(f"""
                    <script>
                    navigator.clipboard.writeText("{js_safe_text}");
                    </script>
                    """, unsafe_allow_html=True)
            
            with col2:
                if st.button("💾 Salva come bozza", key="save_button"):
                    # Aggiorna la bozza salvata
                    email_data.response_draft = edited_response
                    st.session_state.email_cache[os.path.join(email_folder, email_data.filename)] = email_data
                    st.toast("Bozza salvata con successo!")
    else:
        st.info("Seleziona un'email dalla sidebar per visualizzarla e analizzarla.")

def adversarial_evaluator(response: str) -> tuple[bool, str]:
    """
    Valuta la risposta GPT per bias o espressioni discriminatorie.
    Ritorna (True, "OK") se la risposta è accettabile, altrimenti (False, motivo).
    """
    bias_keywords = ['stupido', 'inferiore', 'razza', 'donna', 'uomo']
    for word in bias_keywords:
        if word in response.lower():
            return False, f"Bias rilevato: '{word}'"
    return True, "OK"

def genera_risposta_email(email_anon, clients, config):
    # ...existing code per generare la risposta...
    risposta_gpt, fonti = generate_rag_response(
        clients,
        query=email_anon,
        relevant_docs=[],  # o i chunk trovati
        model_deployment=config.AZURE_CHAT_DEPLOYMENT,
        chat_history=None
    )
    # --- Valutazione avversariale ---
    ok, msg = adversarial_evaluator(risposta_gpt)
    if not ok:
        # Qui puoi decidere se rigenerare, mostrare avviso, loggare, ecc.
        st.warning(f"⚠️ Risposta bloccata per rischio bias: {msg}")
        # Ad esempio: return None, fonti
        return None, fonti
    # ...existing code...
    return risposta_gpt, fonti
