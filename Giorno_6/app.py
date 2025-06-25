import streamlit as st
from G6E1_Rag import SimpleRAG
import os

st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📚 Chat RAG con FAISS + Azure OpenAI")

# Inizializza RAG
rag = SimpleRAG()

# Caricamento PDF
uploaded_file = st.file_uploader("📄 Carica un file PDF", type="pdf")

if uploaded_file:
    with st.spinner("Elaborazione PDF..."):
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        rag.load_pdf(temp_pdf_path)
        st.success("✅ PDF caricato e indicizzato con successo!")

    # Se il PDF è stato caricato con successo, mostra input domanda
    st.subheader("❓ Fai una domanda sul contenuto del PDF")

    question = st.text_input("Inserisci la tua domanda qui:")

    if st.button("🔍 Chiedi") and question.strip():
        with st.spinner("Generazione risposta..."):
            answer = rag.ask(question.strip())
            st.markdown("### 🤖 Risposta")
            st.write(answer)

        # Opzionale: Mostra anche i documenti rilevanti
        with st.expander("📎 Mostra contesto usato"):
            relevant_docs = rag.search(question)
            for i, doc in enumerate(relevant_docs, 1):
                st.markdown(f"**Documento {i}:**\n{doc}\n---")