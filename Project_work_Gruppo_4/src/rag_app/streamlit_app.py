import io
from typing import Dict, Any

import pandas as pd
import streamlit as st

from .constants import APP_TITLE
from .pipeline import RAGPipeline
from .extraction import TemplateExtractor


class StreamlitApp:
    """Gestisce l'interfaccia Streamlit dell'applicazione."""

    def run(self) -> None:  # pragma: no cover - UI heavy
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.markdown(
            f"<h1 style='color:#1F4E79; font-size: 2.5rem; margin-bottom:0.2em;'>{APP_TITLE}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#34495E; font-size:1.1rem;'>"
            "💡 <i>Questa applicazione ti aiuta ad analizzare e sintetizzare bandi caricati in formato PDF o TXT, usando tecniche di Retrieval Augmented Generation (RAG) con Azure OpenAI.</i>"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if "chats" not in st.session_state:
            st.session_state.chats = {}
            st.session_state.active_chat_id = None
            new_id = "chat_1"
            st.session_state.chats[new_id] = {
                "id": new_id,
                "name": "Nuova Chat 1",
                "pipeline": RAGPipeline(),
                "uploaded_file_names": set(),
                "history": [],
                "summaries_cache": {},
            }
            st.session_state.active_chat_id = new_id

        if "info_message" not in st.session_state:
            st.session_state.info_message = ""

        with st.sidebar:
            st.markdown("### 🗂️ Gestione Chat")
            chat_options = {cid: chat["name"] for cid, chat in st.session_state.chats.items()}
            selected_chat_id = st.radio(
                "Seleziona o crea una chat:",
                options=list(chat_options.keys()),
                format_func=lambda x: chat_options.get(x, "Errore Chat"),
                key="chat_selector",
            )
            if selected_chat_id != st.session_state.active_chat_id:
                st.session_state.info_message = ""
            st.session_state.active_chat_id = selected_chat_id
            active_chat = st.session_state.chats[selected_chat_id]

            new_name = st.text_input(
                "✏️ Rinomina chat corrente",
                value=active_chat["name"],
                key=f"rename_chat_{active_chat['id']}",
            )
            if new_name and new_name != active_chat["name"]:
                active_chat["name"] = new_name
                st.session_state.info_message = f"Chat rinominata in '{new_name}'"
                st.rerun()

            st.markdown("---")
            if st.button("➕ Nuova chat", use_container_width=True):
                new_id = f"chat_{len(st.session_state.chats) + 1}"
                default_name = f"Nuova Chat {len(st.session_state.chats) + 1}"
                st.session_state.chats[new_id] = {
                    "id": new_id,
                    "name": default_name,
                    "pipeline": RAGPipeline(),
                    "uploaded_file_names": set(),
                    "history": [],
                    "summaries_cache": {},
                }
                st.session_state.active_chat_id = new_id
                st.session_state.info_message = f"Nuova chat '{default_name}' creata."
                st.rerun()

            st.markdown("### 📥 Caricamento Documenti")
            uploaded_files = st.file_uploader(
                "Trascina qui i tuoi bandi (.pdf, .txt)",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                key=f"file_uploader_{active_chat['id']}",
            )
            if uploaded_files:
                nuovi = [f for f in uploaded_files if f.name not in active_chat["uploaded_file_names"]]
                if nuovi:
                    with st.spinner(f"Elaborazione di {len(nuovi)} file..."):
                        active_chat["pipeline"].add_uploaded_files(nuovi)
                        for f in nuovi:
                            active_chat["uploaded_file_names"].add(f.name)
                    active_chat["summaries_cache"] = {}
                    st.success(st.session_state.info_message)
                    st.rerun()
                elif len(uploaded_files) > 0:
                    st.info("ℹ️ Tutti i file selezionati sono già stati caricati in questa chat.")

            if active_chat["uploaded_file_names"]:
                st.markdown("---")
                st.markdown("### 📂 Documenti Caricati")
                for file_name in sorted(active_chat["uploaded_file_names"]):
                    st.text(f"- {file_name}")
            else:
                st.info("Nessun documento caricato per questa chat.")

            st.markdown("---")
            if st.button("💾 Salva la chat corrente", use_container_width=True):
                filepath = TemplateExtractor.save_chat_to_file(active_chat)
                st.success(f"💾 Chat salvata in: {filepath}")

        if st.session_state.info_message:
            st.info(st.session_state.info_message)
            st.session_state.info_message = ""

        tab_chat, tab_excel, tab_search_idea, tab_handbook = st.tabs(
            ["💬 Chat con i Documenti", "📊 Estrazione Tabella Excel", "🔍 Ricerca Bando per Idea", "📖 Handbook (Riassunti)"]
        )

        with tab_chat:
            st.subheader(f"💬 Interagisci con i documenti di '{active_chat['name']}'")
            chat_container = st.container(height=400, border=True)
            for msg in active_chat["history"]:
                with chat_container.chat_message(msg["role"]):
                    st.write(msg["content"])

            input_widget_key = f"chat_input_actual_value_{active_chat['id']}"
            if input_widget_key not in st.session_state:
                st.session_state[input_widget_key] = ""

            def handle_chat_submit():
                user_input = st.session_state[input_widget_key]
                if not user_input:
                    return
                if not active_chat["pipeline"].documents:
                    st.error("Per favore, carica prima dei documenti per poter fare domande.")
                    st.session_state[input_widget_key] = ""
                    return
                active_chat["history"].append({"role": "user", "content": user_input})
                st.session_state[input_widget_key] = ""
                with st.spinner("Sto elaborando la tua domanda..."):
                    risposta = active_chat["pipeline"].answer_query(user_input)
                    active_chat["history"].append({"role": "assistant", "content": risposta})
                st.rerun()

            with st.form(key=f"chat_form_{active_chat['id']}", clear_on_submit=False):
                st.text_input(
                    "Fai una domanda ai documenti caricati:",
                    key=input_widget_key,
                )
                st.form_submit_button("Invia Domanda", on_click=handle_chat_submit)

        with tab_excel:
            st.subheader("📊 Estrazione Tabella di Sintesi Bandi")
            st.info(
                "Questa sezione estrae informazioni strutturate da tutti i bandi caricati e le presenta in una tabella scaricabile. Puoi anche modificarla qui prima del download."
            )
            if active_chat["pipeline"].documents:
                df_summary_key = f"df_summary_{active_chat['id']}"
                if st.button("📑 Genera e Visualizza Tabella di Sintesi", key=f"generate_excel_{active_chat['id']}"):
                    with st.spinner(
                        "Generazione della tabella in corso. Questo può richiedere del tempo per molti file..."
                    ):
                        extractor = TemplateExtractor(active_chat["pipeline"])
                        df_output = extractor.fill_template_for_all_files()
                        st.session_state[df_summary_key] = df_output
                    st.success("Tabella generata con successo!")
                if df_summary_key in st.session_state:
                    st.markdown("### Tabella di Sintesi (Editabile)")
                    st.info(
                        "Puoi modificare i valori direttamente in questa tabella. Le modifiche saranno incluse nel download Excel."
                    )
                    edited_df = st.data_editor(
                        st.session_state[df_summary_key], use_container_width=True, key=f"data_editor_{active_chat['id']}"
                    )
                    st.session_state[df_summary_key] = edited_df
                    output = io.BytesIO()
                    edited_df.to_excel(output, index=False, engine="openpyxl")
                    st.download_button(
                        label="📥 Scarica Excel (Tabella Corrente)",
                        data=output.getvalue(),
                        file_name=f"bandi_sintesi_{active_chat['name'].replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_excel_{active_chat['id']}",
                    )
            else:
                st.warning("Carica dei documenti nella sidebar per generare la tabella di sintesi.")

        with tab_search_idea:
            st.subheader("🔍 Trova Bandi per la tua Idea Progettuale")
            st.info(
                "Descrivi la tua idea progettuale e cercherò i bandi più pertinenti tra quelli caricati."
            )
            project_idea = st.text_input(
                "Descrivi brevemente la tua idea progettuale (es. 'creazione di una startup'):",
                key=f"project_idea_{active_chat['id']}",
            )
            search_button = st.button("Cerca Bandi Rilevanti", key=f"search_idea_btn_{active_chat['id']}")
            if search_button:
                if not project_idea:
                    st.warning("Per favore, inserisci un'idea progettuale per iniziare la ricerca.")
                elif not active_chat["pipeline"].documents:
                    st.error("Per favore, carica prima dei documenti nella sidebar per poter cercare bandi.")
                elif not active_chat["pipeline"].retriever:
                    st.warning(
                        "Non è stato possibile inizializzare il motore di ricerca dei documenti. Assicurati che i documenti siano stati caricati correttamente."
                    )
                else:
                    with st.spinner(f"Analizzando i bandi per l'idea: '{project_idea}'..."):
                        try:
                            relevant_docs = active_chat["pipeline"].retriever.get_relevant_documents(project_idea)
                            if relevant_docs:
                                files_to_evaluate = {}
                                for doc in relevant_docs:
                                    file_name = doc.metadata.get("file_name", "File Sconosciuto")
                                    files_to_evaluate.setdefault(file_name, []).append(doc.page_content)
                                st.success("Analisi completata. Ecco i bandi potenzialmente rilevanti per la tua idea:")
                                df_summary = st.session_state.get(df_summary_key)
                                for file_name in sorted(files_to_evaluate.keys()):
                                    context_for_eval = "\n\n---\n\n".join(files_to_evaluate[file_name])
                                    suitability_question = f"""
                                Valuta se il seguente bando è adatto o pertinente per l'idea progettuale: "{project_idea}".
                                Basati sul contenuto fornito. Rispondi in modo conciso e fornisci un giudizio chiaro.
                                Contesto del Bando (Estratti):
                                {context_for_eval}
                                """
                                    messages = [
                                        {
                                            "role": "system",
                                            "content": "Sei un esperto nell'analisi di bandi e nella valutazione della loro pertinenza rispetto a specifiche idee progettuali. Rispondi in modo professionale e chiaro.",
                                        },
                                        {"role": "user", "content": suitability_question},
                                    ]
                                    try:
                                        response = active_chat["pipeline"].chat_model.azure_client.chat.completions.create(
                                            model=active_chat["pipeline"].chat_model.model_name,
                                            messages=messages,
                                            max_tokens=250,
                                            temperature=0.2,
                                        )
                                        suitability_answer = response.choices[0].message.content.strip()
                                    except Exception as e:
                                        suitability_answer = f"ERRORE nella valutazione: {e}"
                                        st.error(f"Errore durante la valutazione di idoneità per '{file_name}': {e}")
                                    st.markdown(f"#### 📄 {file_name}")
                                    if df_summary is not None and not df_summary.empty:
                                        row = df_summary[df_summary["File"] == file_name]
                                        if not row.empty:
                                            data = row.iloc[0].to_dict()
                                            key_titolo = "Titolo dell'avviso"
                                            st.markdown(f"**Titolo Avviso:** {data.get(key_titolo, 'N/A')}")
                                            st.markdown(f"**Ente Erogatore:** {data.get('Ente erogatore', 'N/A')}")
                                            st.markdown(f"**Sintesi Bando:** {data.get('Descrizione aggiuntiva', 'N/A')}")
                                            if data.get('Link') and data.get('Link') != 'N/A':
                                                st.markdown(f"**Link:** [{data.get('Link')}]({data.get('Link')})")
                                    else:
                                        st.info("Nessuna sintesi dettagliata disponibile per questo bando. Prova a generare la tabella Excel.")
                                    st.markdown(f"**Valutazione di idoneità per '{project_idea}':**")
                                    st.write(suitability_answer)
                                    st.markdown("---")
                            else:
                                st.info("Nessun bando rilevante trovato per l'idea progettuale fornita. Prova a riformulare la tua idea o caricare più documenti.")
                        except Exception as e:
                            st.error(f"Si è verificato un errore critico durante la ricerca: {e}")
                            st.info("Assicurati che i tuoi documenti siano stati elaborati correttamente e che le API di Azure OpenAI siano accessibili.")

        with tab_handbook:
            st.subheader("📖 Genera Handbook dei Bandi")
            st.info(
                "Questa sezione crea un riassunto narrativo dettagliato per ciascun bando caricato, ideale per una consultazione rapida. I riassunti generati vengono memorizzati per velocizzare le consultazioni successive."
            )
            if active_chat["uploaded_file_names"]:
                selected_file = st.selectbox(
                    "Seleziona un file per generare il riassunto:",
                    options=[""] + sorted(active_chat["uploaded_file_names"]),
                    key=f"select_summary_file_{active_chat['id']}",
                )
                if selected_file:
                    if selected_file in active_chat["summaries_cache"]:
                        st.info(f"Riassunto per '{selected_file}' recuperato dalla cache.")
                        st.markdown(f"#### Riassunto per '{selected_file}'")
                        st.write(active_chat["summaries_cache"][selected_file])
                    if st.button(
                        f"Genera Riassunto per '{selected_file}'",
                        key=f"generate_summary_btn_{active_chat['id']}",
                    ):
                        with st.spinner(f"Generazione del riassunto per '{selected_file}'..."):
                            extractor = TemplateExtractor(active_chat["pipeline"])
                            summary_content = extractor.extract_summary_for_file(selected_file, active_chat)
                            st.markdown(f"#### Riassunto per '{selected_file}'")
                            st.write(summary_content)
                            st.download_button(
                                label="📥 Scarica Riassunto",
                                data=summary_content.encode("utf-8"),
                                file_name=f"riassunto_{selected_file.replace('.', '_')}.txt",
                                mime="text/plain",
                                key=f"download_summary_{active_chat['id']}",
                            )
                else:
                    st.info("Seleziona un file per generare il suo riassunto.")
            else:
                st.warning("Carica dei documenti nella sidebar per generare i riassunti.")


__all__ = ["StreamlitApp"]
