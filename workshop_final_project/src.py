import streamlit as st
from dotenv import load_dotenv

from src.rag_app.pipeline import RAGPipeline
from src.rag_app.template_utils import fill_template_for_all_files

APP_TITLE = "RAG PDF/TXT con Azure OpenAI"


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, page_icon="📄")
    st.title(APP_TITLE)

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
        st.session_state.uploaded_file_names = set()

    uploaded_files = st.file_uploader(
        "Carica uno o più file .pdf o .txt",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        nuovi_file = []
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_file_names:
                nuovi_file.append(file)
                st.session_state.uploaded_file_names.add(file.name)
        if nuovi_file:
            st.session_state.pipeline.add_uploaded_files(nuovi_file)
            st.success(f"{len(nuovi_file)} nuovi file aggiunti.")
        else:
            st.info("Nessun nuovo file da aggiungere.")

    if st.session_state.uploaded_file_names:
        st.markdown("**File caricati:**")
        for f in sorted(st.session_state.uploaded_file_names):
            st.markdown(f"- {f}")

    if st.session_state.pipeline.documents:
        st.markdown("### Compila il template Excel sui bandi caricati")
        if st.button("Estrai dati e scarica Excel"):
            with st.spinner("Estrazione automatica dei dati dai bandi in corso..."):
                df_output = fill_template_for_all_files(st.session_state.pipeline)
                st.success("Estrazione completata! Scarica il file Excel qui sotto.")
                st.dataframe(df_output)
                from io import BytesIO

                output = BytesIO()
                df_output.to_excel(output, index=False, engine="openpyxl")
                st.download_button(
                    label="📥 Scarica Excel compilato",
                    data=output.getvalue(),
                    file_name="bandi_compilati.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    query = st.chat_input("Fai una domanda sui bandi caricati...")
    if query:
        st.chat_message("user").markdown(query)
        risposta = st.session_state.pipeline.answer_query(query)
        st.chat_message("assistant").markdown(risposta)


if __name__ == "__main__":
    main()
