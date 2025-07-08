import os
from pathlib import Path
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
import pandas as pd

from src.rag_app import RAGPipeline
from src.utils import safe_gpt_call
from langchain_community.vectorstores import FAISS

load_dotenv()

APP_TITLE = "RAG PDF/TXT con Azure OpenAI"
TMP_UPLOADS_PATH = Path("tmp_uploads")
TMP_UPLOADS_PATH.mkdir(exist_ok=True)


# ----------------------------
# Template definitions
# ----------------------------
TEMPLATE_FIELDS = [
    "Ente erogatore",
    "Titolo dell'avviso",
    "Descrizione aggiuntiva",
    "Beneficiari",
    "Apertura",
    "Chiusura",
    "Dotazione finanziaria",
    "Contributo",
    "Note",
    "Link",
    "Key Words",
    "Aperto (si/no)",
]

TEMPLATE_QUESTIONS = {
    "Ente erogatore": "Scrivi solo il nome esatto dell’ente erogatore di questo bando, scegliendolo dalle prime tre pagine. Se non lo trovi, deduci quello più probabile dal testo. Solo il nome, nessuna spiegazione. Controlla anche nel titolo del file.",
    "Titolo dell'avviso": "Scrivi solo il titolo ufficiale dell’avviso, così come appare o come puoi dedurlo dalle prime tre pagine. Solo la dicitura, nessuna spiegazione. Controlla anche nel titolo del file.",
    "Descrizione aggiuntiva": "Scrivi una sola frase molto breve (massimo 25 parole) che riassume l’intero bando. Solo la frase, senza spiegazioni. Controlla anche nel titolo del file.",
    "Beneficiari": "Scrivi solo i beneficiari principali di questo bando, anche dedotti dal testo. Solo l’elenco, senza spiegazioni. Controlla anche nel titolo del file.",
    "Apertura": "Scrivi solo la data di apertura (formato GG/MM/AAAA), anche dedotta dal testo se non è esplicitata. Controlla anche nel titolo del file.",
    "Chiusura": "Scrivi solo la data di chiusura (formato GG/MM/AAAA), anche dedotta dal testo se non è esplicitata. Controlla anche nel titolo del file.",
    "Dotazione finanziaria": "Qual è la dotazione finanziaria totale del bando? Scrivi solo la cifra o il valore principale della dotazione finanziaria, anche se devi dedurlo dal testo.",
    "Contributo": "Qual è il contributo previsto per i beneficiari? Scrivi solo la cifra o percentuale principale del contributo previsto, anche se la deduci dal testo.",
    "Note": "Scrivi solo una nota rilevante, anche se la deduci dal testo. Solo la nota, senza spiegazioni.",
    "Link": "Scrivi solo il link (URL) principale trovato nel testo, oppure deducilo se presente in altro modo.",
    "Key Words": "Scrivi solo tre parole chiave, anche dedotte dal testo, separate da virgola e senza spiegazioni.",
    "Aperto (si/no)": "Rispondi solo con 'si' o 'no' se il bando è ancora aperto; deduci la risposta dal testo e dalle date. Nessuna spiegazione.",
}


# ----------------------------
# Utility functions
# ----------------------------

def get_full_texts_per_file(pipeline: RAGPipeline):
    file_to_chunks = {}
    for doc in pipeline.documents:
        file = doc.metadata["file_name"]
        if file not in file_to_chunks:
            file_to_chunks[file] = []
        file_to_chunks[file].append(doc.page_content)
    return {file: "\n".join(chunks) for file, chunks in file_to_chunks.items()}


def save_chat_to_file(active_chat, folder="saved_chats"):
    os.makedirs(folder, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in active_chat["name"])
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Chat: {active_chat['name']}\n")
        f.write(f"Salvata il: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("File caricati:\n")
        for file_name in sorted(active_chat.get("uploaded_file_names", [])):
            f.write(f"- {file_name}\n")
        f.write("\nStoria chat:\n")
        for msg in active_chat.get("history", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            f.write(f"{role.upper()}: {content}\n\n")
    return filepath


def fill_template_for_all_files(pipeline: RAGPipeline, top_k: int = 3) -> pd.DataFrame:
    file_to_docs = {}
    for doc in pipeline.documents:
        file = doc.metadata["file_name"]
        if file not in file_to_docs:
            file_to_docs[file] = []
        file_to_docs[file].append(doc)

    rows = []
    progress = st.progress(0, text="Estrazione in corso...")
    files = list(file_to_docs.keys())
    for idx, file in enumerate(files):
        row = {}
        docs_this_file = file_to_docs[file]
        first_3_chunks = []
        for doc in docs_this_file:
            if len(first_3_chunks) < 3:
                first_3_chunks.append(doc)

        for field in TEMPLATE_FIELDS:
            question = TEMPLATE_QUESTIONS[field]
            if field in ["Ente erogatore", "Titolo dell'avviso"]:
                retriever = FAISS.from_documents(first_3_chunks, pipeline.embedding_wrapper).as_retriever()
                relevant_docs = retriever.get_relevant_documents(question, k=top_k)
            else:
                retriever = FAISS.from_documents(docs_this_file, pipeline.embedding_wrapper).as_retriever()
                relevant_docs = retriever.get_relevant_documents(question, k=top_k)
            context = "\n\n".join([d.page_content for d in relevant_docs])
            if field == "Descrizione aggiuntiva":
                question = "Scrivi una sola frase molto breve (massimo 25 parole) che riassume il bando."
            prompt = f"""Testo selezionato del bando (estratto tramite ricerca semantica):
{context}

Domanda: {question}
Rispondi solo sulla base del testo fornito sopra. Non aggiungere spiegazioni."""
            answer = pipeline.chat_model.ask_about_document(context, prompt)
            row[field] = (answer or "").strip()
        row["File"] = file
        rows.append(row)
        progress.progress((idx + 1) / len(files), text=f"Completato {idx + 1}/{len(files)}")
    progress.empty()
    df = pd.DataFrame(rows)
    return df


# ----------------------------
# Streamlit UI
# ----------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide", initial_sidebar_state="expanded")

    st.markdown(
        f"<h1 style='color:#1F4E79; font-size: 2.5rem; margin-bottom:0.2em;'>{APP_TITLE}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#34495E; font-size:1.1rem;'>"
        "💡 <i>Questa applicazione ti aiuta ad analizzare e sintetizzare bandi caricati in formato PDF o TXT, usando RAG con Azure OpenAI.</i>"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.active_chat_id = None
        new_id = "1"
        st.session_state.chats[new_id] = {
            "id": new_id,
            "name": "Chat 1",
            "pipeline": RAGPipeline(TMP_UPLOADS_PATH),
            "uploaded_file_names": set(),
            "history": [],
            "pending_questions": [],
        }
        st.session_state.active_chat_id = new_id

    with st.sidebar:
        st.markdown("### 🗂️ Gestione Chat")
        chat_names = {cid: chat["name"] for cid, chat in st.session_state.chats.items()}
        selected_chat_id = st.radio(
            "Chat attiva:",
            options=list(chat_names.keys()),
            format_func=lambda x: chat_names.get(x, "➖ Nessuna selezionata"),
        )
        st.session_state.active_chat_id = selected_chat_id
        active_chat = st.session_state.chats[selected_chat_id]

        new_name = st.text_input("✏️ Rinomina chat", value=active_chat["name"])
        if new_name != active_chat["name"]:
            active_chat["name"] = new_name

        if st.button("➕ Nuova chat"):
            new_id = str(len(st.session_state.chats) + 1)
            default_name = f"Chat {new_id}"
            st.session_state.chats[new_id] = {
                "id": new_id,
                "name": default_name,
                "pipeline": RAGPipeline(TMP_UPLOADS_PATH),
                "uploaded_file_names": set(),
                "history": [],
                "pending_questions": [],
            }
            st.session_state.active_chat_id = new_id
            st.rerun()

        uploaded_files = st.file_uploader(
            "⬆️ Carica file per questa chat (.pdf, .txt)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            nuovi_file = [f for f in uploaded_files if f.name not in active_chat["uploaded_file_names"]]
            if nuovi_file:
                active_chat["pipeline"].add_uploaded_files(nuovi_file)
                for f in nuovi_file:
                    active_chat["uploaded_file_names"].add(f.name)
                st.success(f"✅ {len(nuovi_file)} file caricati.")
            else:
                st.info("ℹ️ Nessun nuovo file da caricare.")

        if st.button("💾 Salva chat"):
            filepath = save_chat_to_file(active_chat)
            st.success(f"💾 Salvata in: {filepath}")

        if active_chat["pipeline"].documents:
            st.markdown("### 📋 Estrazione Excel")
            if st.button("📑 Estrai e scarica Excel"):
                with st.spinner("Estrazione in corso..."):
                    df_output = fill_template_for_all_files(active_chat["pipeline"])
                    st.success("✅ Fatto!")
                    output = BytesIO()
                    df_output.to_excel(output, index=False, engine="openpyxl")
                    st.download_button(
                        label="📥 Scarica Excel",
                        data=output.getvalue(),
                        file_name=f"bandi_{active_chat['name'].replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

    st.subheader(f"💬 Chat attiva: {active_chat['name']}")

    with st.form(key=f"form_{active_chat['id']}"):
        user_input = st.text_input("Domanda ai documenti:", key=f"input_{active_chat['id']}")
        submit_button = st.form_submit_button("▶️ Invia")

    if submit_button and user_input:
        active_chat["pending_questions"].append(user_input)

    if active_chat["pending_questions"]:
        question = active_chat["pending_questions"].pop(0)
        with st.spinner("Sto cercando la risposta..."):
            risposta = active_chat["pipeline"].answer_query(question)
        active_chat["history"].append({"role": "user", "content": question})
        active_chat["history"].append({"role": "assistant", "content": risposta})

    for msg in active_chat["history"]:
        role = "Tu" if msg["role"] == "user" else "🤖 Assistente"
        st.markdown(f"**{role}:** {msg['content']}")


if __name__ == "__main__":
    main()
