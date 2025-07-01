import streamlit as st
from config import RagClients, AppConfig
from rag_system import search_documents, generate_rag_response
import re
import time

# --- Funzioni di Supporto ---

def is_memory_update_command(user_input: str) -> bool:
    """Verifica se l'input dell'utente è un comando per aggiornare la memoria."""
    update_patterns = [
        r'\b(ricorda|memorizza|annota) che\b', r'\b(da oggi|d\'ora in poi)\b',
        r'\b(aggiorna|modifica) questa affermazione\b', r'\b(ti informo che|voglio che tu sappia)\b'
    ]
    return any(re.search(pattern, user_input, re.IGNORECASE) for pattern in update_patterns)

def create_new_chat():
    """Crea una nuova chat vuota e la imposta come corrente."""
    chat_id = f"chat_{int(time.time())}"
    st.session_state.chats[chat_id] = {
        "title": "Nuova Conversazione",
        "messages": [],
    }
    st.session_state.current_chat_id = chat_id

def display_message(role, content, sources=None):
    """Visualizza un messaggio nella chat con formattazione appropriata."""
    if role == "user":
        st.chat_message("user").write(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(content)
            # Mostra le fonti se presenti
            if sources and len(sources) > 0:
                with st.expander("🔍 Fonti consultate"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"{i}. {source}")

def display_sidebar_chat_options():
    """Gestisce le opzioni delle chat nella sidebar."""
    st.sidebar.header("Le tue conversazioni")
    
    # Bottone per una nuova chat
    if st.sidebar.button("➕ Nuova conversazione"):
        create_new_chat()
        st.rerun()

    # Visualizza le chat esistenti
    if st.session_state.chats:
        st.sidebar.write("Conversazioni recenti:")
        
        for chat_id, chat in st.session_state.chats.items():
            # Calcola un nome abbreviato per la chat, se necessario
            title = chat.get("title", "Conversazione senza titolo")
            if len(title) > 30:
                display_title = f"{title[:27]}..."
            else:
                display_title = title
                
            # Se la chat è quella corrente, evidenziala
            if chat_id == st.session_state.current_chat_id:
                st.sidebar.markdown(f"**👉 {display_title}**")
            else:
                # Altrimenti mostra un pulsante cliccabile
                if st.sidebar.button(display_title, key=f"chat_btn_{chat_id}"):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
    else:
        st.sidebar.info("Nessuna conversazione salvata.")
        
    st.sidebar.markdown("---")

def display_chat_page(clients: RagClients, config: AppConfig):
    """Visualizza l'interfaccia della pagina di chat."""
    # Inizializza lo stato della sessione se non esiste
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    
    if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
        create_new_chat()
        
    # Mostra le opzioni nella sidebar
    display_sidebar_chat_options()
    
    # Intestazione della pagina
    st.header("💬 Chat Normativa")
    st.write("Poni domande sulla normativa MOCA e ricevi risposte basate sui documenti ufficiali.")
    
    # Recupera la chat corrente
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Visualizza i messaggi della chat corrente
    for message in current_chat["messages"]:
        display_message(message["role"], message["content"], message.get("sources", None))
    
    # Campo di input per la chat
    user_input = st.chat_input("Scrivi il tuo quesito qui...")
    
    if user_input:
        # Aggiunge il messaggio dell'utente alla chat
        current_chat["messages"].append({"role": "user", "content": user_input})
        display_message("user", user_input)
        
        try:
            with st.spinner("🔍 Sto cercando le informazioni pertinenti..."):
                # Ricerca nei documenti
                if not is_memory_update_command(user_input):
                    relevant_docs = search_documents(clients, user_input, config.PINECONE_INDEX_NAME, top_k=5)
                    st.write(f"Ho trovato {len(relevant_docs)} documenti rilevanti.")
                else:
                    relevant_docs = []
                    st.write("Sto aggiornando la memoria del sistema.")
            
            with st.spinner("🤖 Sto elaborando la risposta..."):
                # Genera risposta
                response, sources = generate_rag_response(
                    clients,
                    user_input,
                    relevant_docs,
                    config.AZURE_CHAT_DEPLOYMENT,
                    chat_history=current_chat["messages"][:-1]  # Escludi il messaggio appena aggiunto
                )
                
                st.write("Risposta generata con successo.")
            
            # Aggiunge la risposta alla chat
            current_chat["messages"].append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
            
            # Visualizza la risposta
            display_message("assistant", response, sources)
            
            # Aggiorna il titolo della chat dopo qualche scambio
            if len(current_chat["messages"]) == 2:
                # È il primo scambio, genera un titolo
                try:
                    title_prompt = f"Genera un titolo breve e descrittivo (massimo 6 parole) per una conversazione che inizia con questa domanda: '{user_input}'"
                    title_response = clients.chat_client.chat.completions.create(
                        model=config.AZURE_CHAT_DEPLOYMENT,
                        messages=[{"role": "user", "content": title_prompt}],
                        temperature=0.7,
                        max_tokens=20
                    )
                    current_chat["title"] = title_response.choices[0].message.content.strip('"')
                except Exception as e:
                    current_chat["title"] = user_input[:30] + "..."
                    st.warning(f"Non è stato possibile generare un titolo personalizzato: {e}")
                
        except Exception as e:
            st.error(f"Si è verificato un errore durante l'elaborazione della tua richiesta: {e}")
            # Aggiungi messaggio di errore alla chat
            current_chat["messages"].append({
                "role": "assistant",
                "content": f"Mi dispiace, si è verificato un errore durante l'elaborazione della tua richiesta. Dettagli: {e}"
            })
