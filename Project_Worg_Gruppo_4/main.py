import os
from dotenv import load_dotenv
import streamlit as st

from src.rag_app.pipeline import RAGPipeline

load_dotenv()

DEFAULT_FOLDER_PATH = os.getenv("DEFAULT_FOLDER_PATH", "./Dataset")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # solo per evitare warning su Windows


def main() -> None:
    st.set_page_config(page_title="RAG con Azure OpenAI", page_icon="📄", layout="wide")

    st.title("📄 RAG con Azure OpenAI")
    st.markdown(
        "Interroga i tuoi documenti locali con embeddings **Ada** e risposte da **GPT-4o**.\n"
        "Crea nuove chat e carica file testuali dalla sidebar."
    )

    st.sidebar.header("⚙️ Configurazione")
    folder_path = st.sidebar.text_input("📂 Cartella documenti", value=DEFAULT_FOLDER_PATH)

    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    if st.sidebar.button("➕ Nuova chat"):
        new_chat_id = len(st.session_state.chats)
        try:
            new_pipeline = RAGPipeline(folder_path)
        except Exception as e:
            st.sidebar.error(f"Errore creazione pipeline: {e}")
            return
        st.session_state.chats.append({
            "id": new_chat_id,
            "name": f"Chat #{new_chat_id + 1}",
            "history": [],
            "pipeline": new_pipeline,
            "uploaded_file_names": set(),
        })
        st.session_state.active_chat_id = new_chat_id

    if st.session_state.chats:
        chat_names = [chat["name"] for chat in st.session_state.chats]
        selected_chat_name = st.sidebar.selectbox(
            "🗂️ Seleziona chat",
            chat_names,
            index=st.session_state.active_chat_id or 0,
        )
        st.session_state.active_chat_id = chat_names.index(selected_chat_name)

    uploaded_files = st.sidebar.file_uploader("⬆️ Carica file .txt", type=["txt"], accept_multiple_files=True)

    if st.session_state.active_chat_id is not None and st.session_state.chats:
        chat = st.session_state.chats[st.session_state.active_chat_id]
        pipeline = chat["pipeline"]

        if uploaded_files:
            nuovi_file = []
            for file in uploaded_files:
                if file.name not in chat["uploaded_file_names"]:
                    nuovi_file.append(file)
                    chat["uploaded_file_names"].add(file.name)
            if nuovi_file:
                pipeline.add_uploaded_files(nuovi_file)
                st.sidebar.success(f"{len(nuovi_file)} nuovi file aggiunti.")
            else:
                st.sidebar.info("Nessun nuovo file da aggiungere.")

        if chat["uploaded_file_names"]:
            with st.expander("📄 File caricati"):
                for name in sorted(chat["uploaded_file_names"]):
                    st.markdown(f"- {name}")

        for entry in chat["history"]:
            st.chat_message(entry["role"]).markdown(entry["content"])

        prompt = st.chat_input("Scrivi la tua domanda…")
        if prompt:
            chat["history"].append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)

            risposta = pipeline.answer_query(prompt)
            chat["history"].append({"role": "assistant", "content": risposta})
            st.chat_message("assistant").markdown(risposta)
    else:
        st.info("Crea o seleziona una chat dalla sidebar per iniziare.")


if __name__ == "__main__":
    main()
