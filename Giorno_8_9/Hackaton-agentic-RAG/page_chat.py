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
        "name": "Nuova Chat",
        "messages": [{"role": "assistant", "content": "Salve! Poni la tua prima domanda per iniziare."}],
        "memory_updates": []
    }
    st.session_state.current_chat_id = chat_id
    st.rerun()

def switch_chat(chat_id):
    """Passa a una chat esistente."""
    st.session_state.current_chat_id = chat_id
    st.rerun()

def delete_chat(chat_id_to_delete):
    """Elimina una chat e passa a un'altra disponibile."""
    # Previene l'eliminazione dell'ultima chat
    if len(st.session_state.chats) > 1:
        del st.session_state.chats[chat_id_to_delete]
        # Se la chat eliminata era quella corrente, passa a un'altra
        if st.session_state.current_chat_id == chat_id_to_delete:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        st.rerun()
    else:
        st.toast("Non puoi eliminare l'ultima chat.", icon="⚠️")

# --- UI Principale della Pagina ---

def display_chat_page(clients: RagClients, config: AppConfig):
    """Renderizza l'interfaccia della pagina di chat interattiva con gestione multi-chat."""
    
    # MODIFICATO: Inizializzazione dello stato per la gestione di chat multiple
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        # Crea la prima chat all'avvio
        first_chat_id = f"chat_{int(time.time())}"
        st.session_state.chats[first_chat_id] = {
            "name": "Chat Iniziale",
            "messages": [{"role": "assistant", "content": "Salve! Come posso aiutarti oggi con le normative MOCA?"}],
            "memory_updates": []
        }
        st.session_state.current_chat_id = first_chat_id
    
    # Se per qualche motivo l'ID corrente non è valido, impostane uno valido
    if st.session_state.current_chat_id not in st.session_state.chats:
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

    # Ottieni i dati della chat corrente per un accesso più semplice
    current_chat = st.session_state.chats[st.session_state.current_chat_id]

    # NUOVO: Gestione delle chat nella sidebar
    with st.sidebar:
        st.title("Gestione Chat")
        if st.button("➕ Nuova Chat", use_container_width=True):
            create_new_chat()
        
        st.markdown("---")
        st.subheader("Cronologia")
        
        # Elenca tutte le chat disponibili
        for chat_id, chat_data in st.session_state.chats.items():
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                # Usa un bottone per selezionare la chat
                if st.button(chat_data['name'], key=f"select_{chat_id}", use_container_width=True, type="secondary" if st.session_state.current_chat_id != chat_id else "primary"):
                    switch_chat(chat_id)
            with col2:
                # Bottone per eliminare
                st.button("🗑️", key=f"delete_{chat_id}", on_click=delete_chat, args=(chat_id,), help="Elimina questa chat")

        st.markdown("---")
        st.subheader("Impostazioni RAG")
        top_k_slider = st.slider("Documenti da recuperare", 2, 10, 4, help="Quanti documenti simili verranno cercati per formulare la risposta.")


    # --- Area Principale della Chat ---
    st.header(f"💬 {current_chat['name']}")
    st.markdown("Interroga la base di conoscenza MOCA. Puoi anche fornire aggiornamenti in tempo reale.")

    with st.expander("🧠 Memoria Aggiornamenti Utente", expanded=False):
        if current_chat["memory_updates"]:
            st.info("L'AI seguirà queste istruzioni con la massima priorità:")
            for update in current_chat["memory_updates"]:
                st.markdown(f"- `{update}`")
        else:
            st.write("Nessun aggiornamento manuale fornito in questa sessione.")

    # Visualizzazione dei messaggi della chat corrente
    for message in current_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input dell'utente
    if prompt := st.chat_input("Scrivi qui la tua domanda o il tuo aggiornamento..."):
        current_chat["messages"].append({"role": "user", "content": prompt})
        
        # NUOVO: Assegna un nome alla chat se è la prima domanda
        if len(current_chat["messages"]) == 2 and current_chat["name"] == "Nuova Chat":
            # Prendi le prime 5 parole per il nome
            new_name = " ".join(prompt.split()[:5])
            current_chat["name"] = new_name
            st.rerun()

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if is_memory_update_command(prompt):
                current_chat["memory_updates"].append(prompt)
                response = f"Ho memorizzato la seguente informazione: '{prompt}'. La utilizzerò in questa conversazione."
                current_chat["messages"].append({"role": "assistant", "content": response})
                st.markdown(response)
                st.rerun()
            else:
                with st.spinner("🔍 Sto cercando nei documenti e formulo una risposta..."):
                    search_results = search_documents(prompt, clients, config, top_k=top_k_slider)
                    response = generate_rag_response(
                        query=prompt,
                        search_results=search_results,
                        chat_history=current_chat["messages"],
                        memory_updates=current_chat["memory_updates"],
                        clients=clients,
                        config=config
                    )
                    current_chat["messages"].append({"role": "assistant", "content": response})
                    st.markdown(response)
                    if search_results:
                        with st.expander("📚 Fonti Consultate"):
                            for res in search_results:
                                st.write(f"**Fonte:** {res.source} (Pag: {res.page_number}) - **Rilevanza:** {res.score:.2f}")
                                st.caption(res.content[:250] + "...")
