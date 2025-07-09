import os
import io
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS

from .constants import (
    TEMPLATE_FIELDS,
    TEMPLATE_QUESTIONS,
    SAVED_CHATS_FOLDER,
)
from ..utils import safe_gpt_call


class TemplateExtractor:
    """Utility per estrazioni e salvataggi."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fill_template_for_all_files(self) -> pd.DataFrame:
        file_to_docs = {}
        for doc in self.pipeline.documents:
            file = doc.metadata.get("file_name", "Unknown File")
            file_to_docs.setdefault(file, []).append(doc)

        rows = []
        progress_bar = st.progress(0, text="Estrazione in corso...")
        files = sorted(file_to_docs.keys())
        st.markdown("---")
        st.subheader("Dettagli Estrazione (per Debugging)")

        for idx, file in enumerate(files):
            progress_bar.progress((idx + 1) / len(files), text=f"Estrazione in corso... File: {file} ({idx + 1}/{len(files)})")
            row = {"File": file}
            docs_this_file = file_to_docs[file]
            first_chunks_content = "\n\n".join(d.page_content for d in docs_this_file[:3])
            full_file_content = "\n\n".join(d.page_content for d in docs_this_file)
            file_retriever = None
            if docs_this_file:
                try:
                    temp_vectorstore = FAISS.from_documents(docs_this_file, self.pipeline.embedding_wrapper)
                    file_retriever = temp_vectorstore.as_retriever(search_kwargs={"k": 3})
                except Exception as e:
                    st.warning(
                        f"DEBUG: Impossibile creare retriever per il file {file}: {e}. Le query si baseranno sul full_file_content."
                    )
            for field in TEMPLATE_FIELDS:
                question = TEMPLATE_QUESTIONS[field]
                context_for_query = ""
                if field in ["Ente erogatore", "Titolo dell'avviso", "Descrizione aggiuntiva"]:
                    context_for_query = first_chunks_content if first_chunks_content else full_file_content
                else:
                    if file_retriever:
                        try:
                            relevant_docs_for_field = file_retriever.get_relevant_documents(question)
                            context_for_query = "\n\n".join(d.page_content for d in relevant_docs_for_field)
                        except Exception as e:
                            st.warning(
                                f"DEBUG: Errore durante il recupero documenti per '{field}' in '{file}': {e}. Usando il contenuto completo del file come fallback."
                            )
                            context_for_query = full_file_content
                    else:
                        context_for_query = full_file_content

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Sei un assistente AI specializzato nell'estrazione di informazioni specifiche da documenti. "
                            "Rispondi in modo conciso e attieniti strettamente al formato richiesto dalla domanda. "
                            "Se non trovi l'informazione, rispondi 'N/A'."
                        ),
                    },
                    {"role": "user", "content": f"Testo del documento:\n{context_for_query}\n\nDomanda: {question}"},
                ]
                try:
                    response = safe_gpt_call(
                        self.pipeline.chat_model.azure_client.chat.completions.create,
                        model=self.pipeline.chat_model.model_name,
                        messages=messages,
                        max_tokens=200,
                        temperature=0.0,
                    )
                    answer = response.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"ERRORE: {e}"
                    st.error(f"Errore estrazione per {file} - {field}: {e}")

                row[field] = answer
                with st.expander(f"Debug: {file} - {field}"):
                    st.write(f"**Domanda:** {question}")
                    st.text_area(
                        f"Contesto per '{field}' (primi 500 caratteri)",
                        context_for_query[:500],
                        height=100,
                    )
                    st.write(f"**Risposta Generata:** {answer}")

            rows.append(row)

        progress_bar.empty()
        st.success("✅ Estrazione completata per tutti i bandi!")
        st.markdown("---")
        return pd.DataFrame(rows)

    def get_full_text_for_file(self, file_name: str) -> str:
        full_text_chunks = [
            doc.page_content
            for doc in self.pipeline.documents
            if doc.metadata.get("file_name") == file_name
        ]
        return "\n\n".join(full_text_chunks)

    def extract_summary_for_file(self, file_name: str, active_chat_obj: Dict[str, Any]) -> str:
        if "summaries_cache" not in active_chat_obj:
            active_chat_obj["summaries_cache"] = {}
        if file_name in active_chat_obj["summaries_cache"]:
            return active_chat_obj["summaries_cache"][file_name]

        full_text = self.get_full_text_for_file(file_name)
        if not full_text:
            return "Nessun contenuto trovato per questo file."

        summary_question = (
            "Genera un riassunto conciso e professionale del seguente documento. "
            "Evidenzia i punti chiave come gli obiettivi del bando, i requisiti principali per i beneficiari, "
            "le scadenze importanti, l'ammontare dei finanziamenti disponibili e le modalità di presentazione della domanda. "
            "La lunghezza massima deve essere di 300 parole."
        )

        messages = [
            {
                "role": "system",
                "content": "Sei un assistente AI specializzato nella creazione di riassunti dettagliati e strutturati di bandi.",
            },
            {"role": "user", "content": f"Documento:\n{full_text}\n\nDomanda: {summary_question}"},
        ]

        try:
            response = safe_gpt_call(
                self.pipeline.chat_model.azure_client.chat.completions.create,
                model=self.pipeline.chat_model.model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.3,
                top_p=1.0,
            )
            summary = response.choices[0].message.content.strip()
            active_chat_obj["summaries_cache"][file_name] = summary
            return summary
        except Exception as e:
            st.error(f"Errore durante la generazione del riassunto per {file_name}: {e}")
            return f"Errore durante la generazione del riassunto per {file_name}: {e}"

    @staticmethod
    def save_chat_to_file(active_chat, folder: str = SAVED_CHATS_FOLDER) -> str:
        os.makedirs(folder, exist_ok=True)
        safe_name = "".join(c if c.isalnum() else "_" for c in active_chat["name"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.txt"
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Chat: {active_chat['name']}\n")
            f.write(f"Salvata il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("File caricati:\n")
            for file_name in sorted(active_chat.get("uploaded_file_names", [])):
                f.write(f"- {file_name}\n")
            f.write("\nStoria chat:\n")
            for msg in active_chat.get("history", []):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                f.write(f"{role.upper()}: {content}\n\n")
        return filepath

__all__ = ["TemplateExtractor"]
