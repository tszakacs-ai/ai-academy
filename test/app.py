import os
import io
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import pandas as pd
import time
import json
import re
from datetime import datetime

# Carica le variabili d'ambiente all'avvio dell'applicazione.
# Questo è fondamentale per configurare le credenziali Azure e gli endpoint.
load_dotenv()

# --- CONFIGURAZIONE GLOBALE DELL'APPLICAZIONE ---
APP_TITLE = "Analisi e Estrazione Bandi AI (RAG AI)" 
TMP_UPLOADS_PATH = Path("tmp_uploads") # Cartella temporanea per i file caricati (es. PDF)
TMP_UPLOADS_PATH.mkdir(exist_ok=True) # Crea la cartella se non esiste
SAVED_CHATS_FOLDER = "saved_chats" # Cartella per salvare le conversazioni
Path(SAVED_CHATS_FOLDER).mkdir(exist_ok=True) # Crea la cartella se non esiste

# --- DEFINIZIONI DEI MODELLI AZURE E WRAPPER ---

# Classe base per l'inizializzazione del client Azure AI Project.
# Gestisce il caricamento dell'endpoint e l'autenticazione.
class AIProjectClientDefinition:
    def __init__(self):
        # Recupera l'endpoint del progetto dalle variabili d'ambiente.
        endpoint = os.getenv("PROJECT_ENDPOINT")
        if not endpoint:
            # Se l'endpoint non è definito, mostra un errore critico e ferma l'app Streamlit.
            st.error("ERRORE: PROJECT_ENDPOINT non definito nel file .env. Assicurati che il file .env sia presente e configurato correttamente.")
            st.stop() # Ferma l'esecuzione dell'applicazione.
        self.endpoint = endpoint
        try:
            # Inizializza AIProjectClient con l'endpoint e le credenziali Azure di default.
            self.client = AIProjectClient(
                endpoint=self.endpoint,
                azure_endpoint=self.endpoint, # Duplicato per compatibilità, se necessario.
                credential=DefaultAzureCredential(), # Usa le credenziali Azure configurate nell'ambiente.
            )
        except Exception as e:
            # Cattura qualsiasi errore durante l'inizializzazione e ferma l'app.
            st.exception(f"ERRORE CRITICO: Impossibile inizializzare AIProjectClient. Controlla le tue credenziali Azure e l'endpoint. Dettagli: {e}")
            st.stop() # Ferma l'esecuzione in caso di errore critico.

# Wrapper per il modello di embedding Ada di Azure OpenAI.
# Estende AIProjectClientDefinition per riutilizzare l'inizializzazione del client.
class AdaEmbeddingModel(AIProjectClientDefinition):
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        super().__init__() # Chiama il costruttore della classe base.
        self.model_name = model_name
        # Ottiene il client specifico per le operazioni di embedding da Azure OpenAI.
        self.azure_client = self.client.inference.get_azure_openai_client(api_version="2023-05-15")
    
    # Metodo per generare l'embedding di un singolo testo.
    def embed_text(self, text: str) -> List[float]:
        response = self.azure_client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

# Wrapper per integrare il modello AdaEmbeddingModel con Langchain.
# Langchain richiede un'interfaccia Embeddings specifica.
class LangchainAdaWrapper(Embeddings):
    def __init__(self, ada_model: AdaEmbeddingModel):
        self.ada_model = ada_model # Riferimento al modello AdaEmbeddingModel.
    
    # Genera embedding per una lista di documenti.
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Delega la chiamata al metodo embed_text del modello Ada.
        # Per grandi quantità di testi, si potrebbe ottimizzare con chiamate batch.
        return [self.ada_model.embed_text(text) for text in texts]
    
    # Genera embedding per una singola query.
    def embed_query(self, text: str) -> List[float]:
        return self.ada_model.embed_text(text)
    
# Wrapper per il modello di completamento chat (GPT) di Azure OpenAI.
class ChatCompletionModel(AIProjectClientDefinition):
    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__() # Chiama il costruttore della classe base.
        self.model_name = model_name
        # Ottiene il client specifico per le operazioni di chat completion da Azure OpenAI.
        self.azure_client = self.client.inference.get_azure_openai_client(api_version="2025-01-01-preview")
    
    # Metodo per fare una domanda basata sul contenuto di un documento.
    def ask_about_document(self, content: str, question: str) -> str:
        messages = [
            {"role": "system", "content": "Sei un assistente AI specializzato in analisi di documenti testuali. Fornisci risposte concise e pertinenti basate solo sul contesto fornito. Se non puoi rispondere dal contesto, dì che l'informazione non è disponibile."},
            {"role": "user", "content": f"Documento:\n{content}\n\nDomanda: {question}"},
        ]
        # Esegue la chiamata al modello GPT con gestione dei retry.
        response = safe_gpt_call(
            self.azure_client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7, # Bilancia creatività e fedeltà al testo.
            top_p=1.0,
        )
        # safe_gpt_call ora restituisce un oggetto con choices anche in caso di errore,
        # quindi .choices[0].message.content è sempre sicuro.
        return response.choices[0].message.content

# Classe principale per la pipeline RAG (Retrieval Augmented Generation).
# Gestisce il caricamento, l'indicizzazione dei documenti e la risposta alle query.
class RAGPipeline:
    def __init__(self):
        self.documents = [] # Lista di documenti caricati e chunkizzati.
        # Inizializzazione dei modelli di embedding e chat.
        ada_model = AdaEmbeddingModel() 
        self.embedding_wrapper = LangchainAdaWrapper(ada_model)
        self.vectorstore = None # Vector store (es. FAISS) per la ricerca di similarità.
        self.retriever = None # Retriever per ottenere documenti rilevanti.
        self.chat_model = ChatCompletionModel() # Modello per la generazione della risposta.

    # Aggiunge file caricati alla pipeline, li elabora e aggiorna il vector store.
    def add_uploaded_files(self, uploaded_files):
        new_documents = []
        for file in uploaded_files:
            file_name = file.name
            try:
                if file_name.lower().endswith(".txt"):
                    # Legge il contenuto dei file TXT.
                    content = file.getvalue().decode("utf-8")
                    new_documents.append(Document(page_content=content, metadata={"file_name": file_name}))
                elif file_name.lower().endswith(".pdf"):
                    # Salva temporaneamente il PDF per permettere a PyPDFLoader di leggerlo.
                    tmp_path = TMP_UPLOADS_PATH / file_name
                    with open(tmp_path, "wb") as f:
                        f.write(file.getvalue())
                    loader = PyPDFLoader(str(tmp_path))
                    pages = loader.load() # Carica le pagine dal PDF.
                    for page in pages:
                        # Aggiunge il nome del file ai metadati di ogni pagina.
                        page.metadata["file_name"] = file_name 
                    
                    # Suddivide i documenti in chunk più piccoli per il RAG.
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = splitter.split_documents(pages)
                    new_documents.extend(docs)
                    os.remove(tmp_path) # Rimuove il file temporaneo.
                else:
                    st.warning(f"Tipo di file non supportato: {file_name}. Saltato.")
            except Exception as e:
                st.error(f"Errore durante l'elaborazione del file '{file_name}': {e}")
        
        if new_documents:
            self.documents.extend(new_documents) # Aggiunge i nuovi documenti.
            self._build_vectorstore() # Ricostruisce il vector store.
            st.session_state.info_message = f"✅ {len(uploaded_files)} file elaborati e aggiunti."
        else:
            st.session_state.info_message = "ℹ️ Nessun nuovo file valido da elaborare."

    # Costruisce o ricostruisce il vector store FAISS e il retriever.
    def _build_vectorstore(self):
        if self.documents:
            try:
                # Crea un vector store FAISS dai documenti usando l'embedding wrapper.
                self.vectorstore = FAISS.from_documents(self.documents, embedding=self.embedding_wrapper)
                self.retriever = self.vectorstore.as_retriever() # Inizializza il retriever.
            except Exception as e:
                st.exception(f"ERRORE: Impossibile creare il Vectorstore. Controlla il modello di embedding e i documenti. Dettagli: {e}")
                self.vectorstore = None
                self.retriever = None
        else:
            self.vectorstore = None
            self.retriever = None

    # Risponde a una query dell'utente recuperando documenti rilevanti e generando una risposta.
    def answer_query(self, query: str) -> str:
        if not self.retriever:
            return "Nessun documento caricato o indicizzato per la ricerca. Carica prima dei file."
        
        try:
            # Recupera i documenti più simili alla query.
            docs_simili = self.retriever.get_relevant_documents(query)
        except Exception as e:
            st.error(f"Errore durante il recupero dei documenti: {e}")
            return f"Spiacente, si è verificato un errore durante la ricerca nei documenti: {e}"
        
        if not docs_simili:
            return "Nessun documento rilevante trovato per la tua domanda."
        
        # Concatena il contenuto dei documenti rilevanti per formare il contesto.
        context_content = "\n\n---\n\n".join([doc.page_content for doc in docs_simili])
        
        # Cerca link nel contesto per aggiungere al feedback.
        link_found = ""
        link_match = re.search(r'(https?://\S+)', context_content)
        if link_match:
            link_found = f"\n\n[Link Rilevante]({link_match.group(0)})"

        # Genera la risposta usando il modello di chat con il contesto.
        risposta = self.chat_model.ask_about_document(context_content, query)

        # Identifica i file sorgente da cui provengono i documenti rilevanti.
        source_files = ", ".join(list(set([doc.metadata.get("file_name", "N/A") for doc in docs_simili])))
        
        return f"{risposta}\n\n*Fonte/i: {source_files}*{link_found}"


# --- FUNZIONI HELPER ---

# Funzione robusta per le chiamate API GPT con gestione dei retry e degli errori.
def safe_gpt_call(func, *args, max_retries=5, wait_seconds=60, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:
                # Gestisce il rate limit (troppe richieste).
                st.warning(f"Rate limit Azure superato! Aspetto {wait_seconds} secondi... (Tentativo {attempt+1}/{max_retries})")
                time.sleep(wait_seconds)
            else:
                # Gestisce altri errori API.
                st.error(f"Errore durante la chiamata API di Azure OpenAI: {e}")
                # Restituisce un oggetto fittizio per non interrompere il flusso dell'applicazione,
                # permettendo di visualizzare l'errore senza crash.
                return type('obj', (object,), {'choices': [{'message': {'content': f"Errore API: {e}"}}]})() 
    # Se tutti i retry falliscono.
    st.error("Superato il numero massimo di retry per il rate limit.")
    return type('obj', (object,), {'choices': [{'message': {'content': "Errore: Superato il numero massimo di tentativi per il rate limit."}}]})()


# --- DOMANDE PER L'ESTRAZIONE STRUTTURATA (Tabella Excel) ---

# Campi del template per l'estrazione delle informazioni.
TEMPLATE_FIELDS = [
    "Ente erogatore", "Titolo dell'avviso", "Descrizione aggiuntiva", "Beneficiari",
    "Apertura", "Chiusura", "Dotazione finanziaria", "Contributo", "Note",
    "Link", "Key Words", "Aperto (si/no)"
]
 
# Domande specifiche per ogni campo da estrarre.
TEMPLATE_QUESTIONS = {
    "Ente erogatore": "Scrivi solo il nome esatto dell’ente erogatore di questo bando, scegliendolo dalle prime tre pagine. Se non lo trovi, deduci quello più probabile dal testo. Solo il nome, nessuna spiegazione.",
    "Titolo dell'avviso": "Scrivi solo il titolo ufficiale dell’avviso, così come appare o come puoi dedurlo dalle prime tre pagine. Solo la dicitura, nessuna spiegazione.",
    "Descrizione aggiuntiva": "Scrivi una sola frase molto breve (massimo 25 parole) che riassume l’intero bando. Solo la frase, senza spiegazioni.",
    "Beneficiari": "Scrivi solo i beneficiari principali di questo bando, anche dedotti dal testo. Solo l’elenco, senza spiegazioni.",
    "Apertura": "Scrivi solo la data di apertura (formato GG/MM/AAAA), anche dedotta dal testo se non è esplicitata.",
    "Chiusura": "Scrivi solo la data di chiusura (formato GG/MM/AAAA), anche dedotta dal testo se non è esplicitata.",
    "Dotazione finanziaria": "Qual è la dotazione finanziaria totale del bando? Scrivi solo la cifra o il valore principale della dotazione finanziaria, anche se devi dedurlo dal testo.",
    "Contributo": "Qual è il contributo previsto per i beneficiari? Scrivi solo la cifra o percentuale principale del contributo previsto, anche se la deduci dal testo.",
    "Note": "Scrivi solo una nota rilevante, anche se la deduci dal testo. Solo la nota, senza spiegazioni.",
    "Link": "Scrivi solo il link (URL) principale trovato nel testo, oppure deducilo se presente in altro modo.",
    "Key Words": "Scrivi solo tre parole chiave, anche dedotte dal testo, separate da virgola e senza spiegazioni.",
    "Aperto (si/no)": "Rispondi solo con 'si' o 'no' se il bando è ancora aperto; deduci la risposta dal testo e dalle date. Nessuna spiegazione."
}

# Funzione per compilare il template di estrazione per tutti i file caricati.
def fill_template_for_all_files(pipeline: RAGPipeline) -> pd.DataFrame:
    file_to_docs = {}
    # Raggruppa i documenti per nome file.
    for doc in pipeline.documents:
        file = doc.metadata.get("file_name", "Unknown File")
        if file not in file_to_docs:
            file_to_docs[file] = []
        file_to_docs[file].append(doc)
    
    rows = []
    progress_text = "Estrazione in corso..."
    # Barra di progresso per l'estrazione.
    progress_bar = st.progress(0, text=progress_text)
    files = sorted(list(file_to_docs.keys()))

    st.markdown("---")
    st.subheader("Dettagli Estrazione (per Debugging)")

    for idx, file in enumerate(files):
        current_progress = (idx + 1) / len(files)
        progress_bar.progress(current_progress, text=f"{progress_text} File: {file} ({idx + 1}/{len(files)})")

        row = {"File": file}
        docs_this_file = file_to_docs[file]
        
        # Prende i primi chunk per le domande iniziali (ente, titolo).
        first_chunks_content = "\n\n".join([d.page_content for d in docs_this_file[:3]])
        # Contenuto completo del file per le domande che richiedono più contesto.
        full_file_content = "\n\n".join([d.page_content for d in docs_this_file])

        file_retriever = None
        if docs_this_file:
            try:
                # Crea un retriever temporaneo per il singolo file.
                temp_vectorstore = FAISS.from_documents(docs_this_file, pipeline.embedding_wrapper)
                file_retriever = temp_vectorstore.as_retriever(search_kwargs={"k": 3})
            except Exception as e:
                st.warning(f"DEBUG: Impossibile creare retriever per il file {file}: {e}. Le query si baseranno sul full_file_content.")

        for field in TEMPLATE_FIELDS:
            question = TEMPLATE_QUESTIONS[field]
            
            context_for_query = ""
            if field in ["Ente erogatore", "Titolo dell'avviso", "Descrizione aggiuntiva"]:
                # Per questi campi, usa i primi chunk o l'intero documento se necessario.
                context_for_query = first_chunks_content if len(first_chunks_content) > 0 else full_file_content
            else:
                if file_retriever:
                    try:
                        # Usa il retriever per trovare i documenti più rilevanti per la domanda specifica.
                        relevant_docs_for_field = file_retriever.get_relevant_documents(question)
                        context_for_query = "\n\n".join([d.page_content for d in relevant_docs_for_field])
                    except Exception as e:
                        st.warning(f"DEBUG: Errore durante il recupero documenti per '{field}' in '{file}': {e}. Usando il contenuto completo del file come fallback.")
                        context_for_query = full_file_content 
                else:
                    context_for_query = full_file_content

            messages = [
                {"role": "system", "content": "Sei un assistente AI specializzato nell'estrazione di informazioni specifiche da documenti. Rispondi in modo conciso e attieniti strettamente al formato richiesto dalla domanda. Se non trovi l'informazione, rispondi 'N/A'."},
                {"role": "user", "content": f"Testo del documento:\n{context_for_query}\n\nDomanda: {question}"},
            ]
            
            try:
                # Esegue la chiamata al modello di chat per estrarre l'informazione.
                response = safe_gpt_call(
                    pipeline.chat_model.azure_client.chat.completions.create,
                    model=pipeline.chat_model.model_name,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.0, # Temperatura bassa per risposte precise e meno creative.
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"ERRORE: {e}"
                st.error(f"Errore estrazione per {file} - {field}: {e}")

            row[field] = answer
            
            # Sezione per il debugging, espandibile.
            with st.expander(f"Debug: {file} - {field}"):
                st.write(f"**Domanda:** {question}")
                st.text_area(f"Contesto per '{field}' (primi 500 caratteri)", context_for_query[:500], height=100)
                st.write(f"**Risposta Generata:** {answer}")
                
        rows.append(row)
        
    progress_bar.empty() # Rimuove la barra di progresso.
    st.success("✅ Estrazione completata per tutti i bandi!")
    st.markdown("---")
    df = pd.DataFrame(rows)
    return df

# Recupera il testo completo di un file specifico.
def get_full_text_for_file(pipeline: RAGPipeline, file_name: str) -> str:
    full_text_chunks = [doc.page_content for doc in pipeline.documents if doc.metadata.get("file_name") == file_name]
    return "\n\n".join(full_text_chunks)

# Estrae un riassunto per un file specifico.
def extract_summary_for_file(pipeline: RAGPipeline, file_name: str, active_chat_obj: Dict[str, Any]) -> str:
    # Cerca il riassunto nella cache della sessione prima di generarlo.
    # active_chat_obj è già il riferimento a st.session_state.chats[selected_chat_id]
    if "summaries_cache" not in active_chat_obj:
        active_chat_obj["summaries_cache"] = {}

    if file_name in active_chat_obj["summaries_cache"]:
        return active_chat_obj["summaries_cache"][file_name]

    full_text = get_full_text_for_file(pipeline, file_name)
    if not full_text:
        return "Nessun contenuto trovato per questo file."

    summary_question = """
    Genera un riassunto conciso e professionale del seguente documento. Evidenzia i punti chiave come gli obiettivi del bando, i requisiti principali per i beneficiari, le scadenze importanti, l'ammontare dei finanziamenti disponibili e le modalità di presentazione della domanda. La lunghezza massima deve essere di 300 parole.
    """
    
    messages = [
        {"role": "system", "content": "Sei un assistente AI specializzato nella creazione di riassunti dettagliati e strutturati di bandi."},
        {"role": "user", "content": f"Documento:\n{full_text}\n\nDomanda: {summary_question}"},
    ]

    try:
        # Esegue la chiamata al modello di chat per generare il riassunto.
        response = safe_gpt_call(
            pipeline.chat_model.azure_client.chat.completions.create,
            model=pipeline.chat_model.model_name,
            messages=messages,
            max_tokens=400, # Massima lunghezza per il riassunto.
            temperature=0.3, # Bassa temperatura per riassunti più fedeli al testo.
            top_p=1.0,
        )
        summary = response.choices[0].message.content.strip()
        # Salva il riassunto nella cache della sessione.
        active_chat_obj["summaries_cache"][file_name] = summary
        return summary
    except Exception as e:
        st.error(f"Errore durante la generazione del riassunto per {file_name}: {e}")
        return f"Errore durante la generazione del riassunto per {file_name}: {e}"

# Funzione per salvare la chat in un file di testo.
def save_chat_to_file(active_chat, folder="saved_chats"):
    """
    Salva la chat attiva in un file di testo nella cartella specificata.
    Il nome del file è basato sul nome della chat e un timestamp.
    """
    os.makedirs(folder, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in active_chat["name"]) # Rende il nome sicuro per il file.
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

# --- FUNZIONE PRINCIPALE DELL'APPLICAZIONE STREAMLIT ---
def main():
    # Configurazione della pagina Streamlit.
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Titolo e sottotitolo dell'applicazione con stile Markdown/HTML.
    st.markdown(
        f"<h1 style='color:#1F4E79; font-size: 2.5rem; margin-bottom:0.2em;'>{APP_TITLE}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#34495E; font-size:1.1rem;'>"
        "💡 <i>Questa applicazione ti aiuta ad analizzare e sintetizzare bandi caricati in formato PDF o TXT, usando tecniche di Retrieval Augmented Generation (RAG) con Azure OpenAI.</i>"
        "</p>", unsafe_allow_html=True
    )
    st.markdown("---")

    # --- SEZIONE INFORMAZIONI SULL'APPLICAZIONE E FUNZIONAMENTO (Spostata qui) ---
    with st.expander("ℹ️ Informazioni sull'Applicazione e Guida all'Uso"):
        st.markdown("Questa applicazione è stata concepita per semplificare l'analisi e l'estrazione di informazioni cruciali da documenti complessi come i bandi di finanziamento. Utilizza tecniche avanzate di Intelligenza Artificiale, in particolare la *Retrieval Augmented Generation (RAG)*, alimentata da modelli Azure OpenAI.")

        st.markdown("### Come Funziona 'Analisi e Estrazione Bandi AI':")
        st.markdown("""
        1.  **Caricamento Documenti (Sidebar):**
            * Nella sidebar a sinistra, puoi iniziare una "Nuova chat" o selezionarne una esistente.
            * Utilizza la sezione "Caricamento Documenti" per trascinare e caricare i tuoi file (PDF o TXT). L'applicazione elaborerà automaticamente i documenti, suddividendoli in parti più piccole e creando un indice ricercabile (Vector Store). Questo processo è fondamentale per la funzionalità RAG.
            * I nomi dei file caricati per la chat corrente saranno visibili nella sezione "Documenti Caricati".
            * Puoi "Salvare la chat corrente" in qualsiasi momento per conservarne la cronologia e i riferimenti ai file.
        
        2.  **💬 Chat con i Documenti:**
            * Questa è la modalità di interazione principale. Dopo aver caricato i documenti, puoi fare domande in linguaggio naturale relative al loro contenuto.
            * Il sistema recupererà le sezioni più rilevanti dai tuoi documenti e userà queste informazioni per formulare risposte precise e contestualizzate.
            * La chat manterrà una cronologia delle tue domande e delle risposte dell'AI.
        
        3.  **📊 Estrazione Tabella Excel:**
            * Questa sezione è pensata per l'estrazione strutturata di dati. Cliccando su "Genera e Visualizza Tabella di Sintesi", l'AI analizzerà ogni bando caricato e compilerà una tabella con campi predefiniti (es. Ente erogatore, Scadenze, Contributo, etc.).
            * La tabella è editabile direttamente nell'applicazione, permettendoti di correggere o integrare le informazioni estratte.
            * Una volta soddisfatto, puoi scaricare la tabella come file Excel.
        
        4.  **🔍 Ricerca Bando per Idea:**
            * Hai un'idea progettuale ma non sai quale bando si adatta meglio? Inserisci una breve descrizione della tua idea qui.
            * L'AI cercherà tra tutti i documenti caricati quelli più allineati alla tua descrizione, **fornendo una valutazione esplicita sull'idoneità di ciascun bando rilevante** rispetto alla tua idea.
        
        5.  **📖 Handbook (Riassunti):**
            * Seleziona un singolo file dalla lista.
            * Clicca su "Genera Riassunto" per ottenere una sintesi narrativa più estesa e professionale del documento selezionato, utile per una comprensione rapida dei punti salienti.
            * **Per velocizzare:** I riassunti generati vengono automaticamente memorizzati nella sessione corrente. Se richiedi lo stesso riassunto una seconda volta, verrà recuperato istantaneamente dalla memoria, senza dover richiamare il modello AI. I riassunti vengono rigenerati (e la cache svuotata) solo quando carichi nuovi documenti, per assicurare che siano sempre basati sull'ultima versione dei dati.
            * Il riassunto generato può essere scaricato come file di testo.
        """)

        st.markdown("### Dettagli Tecnici:")
        st.markdown("""
        * **Motore RAG:** Utilizza Langchain per la gestione dei documenti e FAISS per la ricerca di similarità vettoriale.
        * **Modelli AI:** Sfrutta le capacità di Azure OpenAI per gli embedding (`text-embedding-ada-002`) e per la generazione del linguaggio (`gpt-4o`).
        * **Gestione Errori:** Include meccanismi di retry per le chiamate API di Azure per gestire problemi temporanei come i rate limit.
        """)

        st.markdown("---")
        st.markdown("AI Academy Project -- Gruppo n.4") 
        st.markdown("- Joe Luigi Scaglione") 
        st.markdown("- Daniel Craciun")
        st.markdown("- Matteo Loi")
        st.markdown("- Giovanni Zagaria")
        st.markdown("- Tania Pia Aloe")
        st.markdown("---")


    # Inizializzazione dello stato della sessione per le chat.
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.active_chat_id = None
        new_id = "chat_1"
        st.session_state.chats[new_id] = {
            "id": new_id,
            "name": "Nuova Chat 1",
            "pipeline": RAGPipeline(), # Ogni chat ha la propria pipeline RAG.
            "uploaded_file_names": set(),
            "history": [],
            "summaries_cache": {} # Cache per i riassunti specifici di questa chat
        }
        st.session_state.active_chat_id = new_id
    
    # Messaggio informativo globale.
    if "info_message" not in st.session_state:
        st.session_state.info_message = ""

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### 🗂️ Gestione Chat")

        chat_options = {cid: chat["name"] for cid, chat in st.session_state.chats.items()}
        selected_chat_id = st.radio(
            "Seleziona o crea una chat:",
            options=list(chat_options.keys()),
            format_func=lambda x: chat_options.get(x, "Errore Chat"),
            key="chat_selector"
        )
        if selected_chat_id != st.session_state.active_chat_id:
            st.session_state.info_message = "" # Resetta il messaggio informativo al cambio chat.
        st.session_state.active_chat_id = selected_chat_id
        active_chat = st.session_state.chats[selected_chat_id]

        # Campo per rinominare la chat corrente.
        new_name = st.text_input("✏️ Rinomina chat corrente", value=active_chat["name"], key=f"rename_chat_{active_chat['id']}")
        if new_name and new_name != active_chat["name"]:
            active_chat["name"] = new_name
            st.session_state.info_message = f"Chat rinominata in '{new_name}'"
            st.rerun() # Forza il refresh per aggiornare il nome nella radio button.

        st.markdown("---")

        # Pulsante per creare una nuova chat.
        if st.button("➕ Nuova chat", use_container_width=True):
            new_id = f"chat_{len(st.session_state.chats) + 1}"
            default_name = f"Nuova Chat {len(st.session_state.chats) + 1}"
            st.session_state.chats[new_id] = {
                "id": new_id,
                "name": default_name,
                "pipeline": RAGPipeline(), # Nuova pipeline per la nuova chat.
                "uploaded_file_names": set(),
                "history": [],
                "summaries_cache": {} # Nuova cache per la nuova chat
            }
            st.session_state.active_chat_id = new_id
            st.session_state.info_message = f"Nuova chat '{default_name}' creata."
            st.rerun() # Forza il refresh per selezionare la nuova chat.

        st.markdown("### 📥 Caricamento Documenti")
        uploaded_files = st.file_uploader(
            "Trascina qui i tuoi bandi (.pdf, .txt)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"file_uploader_{active_chat['id']}" # Chiave unica per il file uploader per chat.
        )
        if uploaded_files:
            # Filtra solo i file che non sono già stati caricati in questa chat.
            nuovi_file_da_elaborare = [f for f in uploaded_files if f.name not in active_chat["uploaded_file_names"]]
            if nuovi_file_da_elaborare:
                with st.spinner(f"Elaborazione di {len(nuovi_file_da_elaborare)} file..."):
                    active_chat["pipeline"].add_uploaded_files(nuovi_file_da_elaborare)
                    for f in nuovi_file_da_elaborare:
                        active_chat["uploaded_file_names"].add(f.name)
                # Resetta la cache dei riassunti ogni volta che vengono caricati nuovi documenti
                # per evitare riassunti obsoleti.
                active_chat["summaries_cache"] = {} 
                st.success(st.session_state.info_message)
                st.rerun() # Forzo un rerun per aggiornare lo stato e i documenti visualizzati.
            elif len(uploaded_files) > 0:
                st.info("ℹ️ Tutti i file selezionati sono già stati caricati in questa chat.")
        
        # Mostra i documenti già caricati nella chat corrente.
        if active_chat["uploaded_file_names"]:
            st.markdown("---")
            st.markdown("### 📂 Documenti Caricati")
            for file_name in sorted(list(active_chat["uploaded_file_names"])):
                st.text(f"- {file_name}")
        else:
            st.info("Nessun documento caricato per questa chat.")

        st.markdown("---")

        # Pulsante per salvare la chat corrente.
        if st.button("💾 Salva la chat corrente", use_container_width=True):
            filepath = save_chat_to_file(active_chat)
            st.success(f"💾 Chat salvata in: {filepath}")

    # Mostra messaggi informativi globali.
    if st.session_state.info_message:
        st.info(st.session_state.info_message)
        st.session_state.info_message = ""

    # --- TAB DEL CONTENUTO PRINCIPALE ---
    tab_chat, tab_excel, tab_search_idea, tab_handbook = st.tabs([
        "💬 Chat con i Documenti",
        "📊 Estrazione Tabella Excel",
        "🔍 Ricerca Bando per Idea",
        "📖 Handbook (Riassunti)"
    ])

    # --- TAB: CHAT CON I DOCUMENTI ---
    with tab_chat:
        st.subheader(f"💬 Interagisci con i documenti di '{active_chat['name']}'")
        
        # Contenitore per la visualizzazione della cronologia della chat.
        chat_container = st.container(height=400, border=True)
        for msg in active_chat["history"]:
            with chat_container.chat_message(msg["role"]):
                st.write(msg["content"])

        # Inizializza l'input della chat nel session_state per controllarne il valore.
        input_widget_key = f"chat_input_actual_value_{active_chat['id']}"
        if input_widget_key not in st.session_state:
            st.session_state[input_widget_key] = ""

        # Callback function per gestire l'invio del messaggio.
        def handle_chat_submit():
            user_input = st.session_state[input_widget_key]
            
            if not user_input: 
                return # Non fare nulla se l'input è vuoto.

            if not active_chat["pipeline"].documents:
                st.error("Per favore, carica prima dei documenti per poter fare domande.")
                st.session_state[input_widget_key] = "" # Pulisci l'input anche se non ci sono documenti.
                return 

            # Aggiungi la domanda dell'utente alla cronologia.
            active_chat["history"].append({"role": "user", "content": user_input})
            
            # Resetta l'input della chat *prima* di generare la risposta per una migliore UX.
            st.session_state[input_widget_key] = "" 

            # Genera la risposta.
            with st.spinner("Sto elaborando la tua domanda..."):
                risposta = active_chat["pipeline"].answer_query(user_input)
                active_chat["history"].append({"role": "assistant", "content": risposta})
            
            # Forziamo un rerun dopo che la history è stata aggiornata.
            # Questo garantisce che Streamlit ridisegni il chat_container con i nuovi messaggi.
            st.rerun() 

        # Form per l'input utente.
        with st.form(key=f"chat_form_{active_chat['id']}", clear_on_submit=False): 
            user_input_widget = st.text_input(
                "Fai una domanda ai documenti caricati:",
                key=input_widget_key, 
                on_change=None # on_change non è necessario con un pulsante di submit.
            )
            st.form_submit_button(
                "Invia Domanda",
                on_click=handle_chat_submit 
            )
        
    # --- TAB: ESTRAZIONE TABELLA EXCEL ---
    with tab_excel:
        st.subheader("📊 Estrazione Tabella di Sintesi Bandi")
        st.info("Questa sezione estrae informazioni strutturate da tutti i bandi caricati e le presenta in una tabella scaricabile. Puoi anche modificarla qui prima del download.")

        if active_chat["pipeline"].documents:
            # Bug fix: Aggiungere una chiave specifica all'interno della session_state per il DataFrame
            # per assicurarsi che non ci siano conflitti tra le chat.
            df_summary_key = f"df_summary_{active_chat['id']}"

            if st.button("📑 Genera e Visualizza Tabella di Sintesi", key=f"generate_excel_{active_chat['id']}"):
                with st.spinner("Generazione della tabella in corso. Questo può richiedere del tempo per molti file..."):
                    df_output = fill_template_for_all_files(active_chat["pipeline"])
                    st.session_state[df_summary_key] = df_output
                st.success("Tabella generata con successo!")
            
            if df_summary_key in st.session_state:
                st.markdown("### Tabella di Sintesi (Editabile)")
                st.info("Puoi modificare i valori direttamente in questa tabella. Le modifiche saranno incluse nel download Excel.")
                # Bug fix: Assicurarsi che st.data_editor lavori sul DataFrame corretto nello stato della sessione.
                edited_df = st.data_editor(st.session_state[df_summary_key], use_container_width=True, key=f"data_editor_{active_chat['id']}")
                st.session_state[df_summary_key] = edited_df # Aggiorna il DataFrame nello stato della sessione dopo la modifica.

                output = io.BytesIO()
                edited_df.to_excel(output, index=False, engine="openpyxl")
                st.download_button(
                    label="📥 Scarica Excel (Tabella Corrente)",
                    data=output.getvalue(),
                    file_name=f"bandi_sintesi_{active_chat['name'].replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{active_chat['id']}"
                )
        else:
            st.warning("Carica dei documenti nella sidebar per generare la tabella di sintesi.")

    # --- TAB: RICERCA BANDO PER IDEA ---
    with tab_search_idea:
        st.subheader("🔍 Trova Bandi per la tua Idea Progettuale")
        st.info("Descrivi la tua idea progettuale e cercherò i bandi più pertinenti tra quelli caricati.")

        project_idea = st.text_input("Descrivi brevemente la tua idea progettuale (es. 'creazione di una startup'):", key=f"project_idea_{active_chat['id']}")
        search_button = st.button("Cerca Bandi Rilevanti", key=f"search_idea_btn_{active_chat['id']}")

        if search_button: 
            if not project_idea:
                st.warning("Per favore, inserisci un'idea progettuale per iniziare la ricerca.")
            elif not active_chat["pipeline"].documents:
                st.error("Per favore, carica prima dei documenti nella sidebar per poter cercare bandi.")
            elif not active_chat["pipeline"].retriever: # Aggiunto controllo per il retriever.
                st.warning("Non è stato possibile inizializzare il motore di ricerca dei documenti. Assicurati che i documenti siano stati caricati correttamente.")
            else:
                with st.spinner(f"Analizzando i bandi per l'idea: '{project_idea}'..."):
                    try:
                        # Ottieni i documenti più rilevanti. Aumentiamo k per avere più contesto.
                        relevant_docs_for_idea = active_chat["pipeline"].retriever.get_relevant_documents(project_idea)
                        
                        if relevant_docs_for_idea:
                            # Raggruppa i documenti rilevanti per file
                            files_to_evaluate = {}
                            for doc in relevant_docs_for_idea:
                                file_name = doc.metadata.get("file_name", "File Sconosciuto")
                                if file_name not in files_to_evaluate:
                                    files_to_evaluate[file_name] = []
                                files_to_evaluate[file_name].append(doc.page_content)
                            
                            st.success(f"Analisi completata. Ecco i bandi potenzialmente rilevanti per la tua idea:")
                            
                            df_summary = st.session_state.get(f"df_summary_{active_chat['id']}")

                            for file_name in sorted(files_to_evaluate.keys()):
                                # Costruisci un contesto più ampio per il modello, specifico per questo file
                                context_for_evaluation = "\n\n---\n\n".join(files_to_evaluate[file_name])
                                
                                # Prompt più specifico per la valutazione di idoneità
                                suitability_question = f"""
                                Valuta se il seguente bando è adatto o pertinente per l'idea progettuale: "{project_idea}".
                                Basati sul contenuto fornito. Rispondi in modo conciso e fornisci un giudizio chiaro (Es: "Sì, il bando è molto adatto per...", "Parzialmente adatto, con focus su...", "No, il bando non sembra pertinente per...").
                                Se rilevante, menziona brevemente perché.
                                Contesto del Bando (Estratti):
                                {context_for_evaluation}
                                """
                                messages = [
                                    {"role": "system", "content": "Sei un esperto nell'analisi di bandi e nella valutazione della loro pertinenza rispetto a specifiche idee progettuali. Rispondi in modo professionale e chiaro."},
                                    {"role": "user", "content": suitability_question},
                                ]

                                try:
                                    response = safe_gpt_call(
                                        active_chat["pipeline"].chat_model.azure_client.chat.completions.create,
                                        model=active_chat["pipeline"].chat_model.model_name,
                                        messages=messages,
                                        max_tokens=250,
                                        temperature=0.2, # Manteniamo la temperatura bassa per risposte fattuali
                                    )
                                    suitability_answer = response.choices[0].message.content.strip()
                                except Exception as e:
                                    suitability_answer = f"ERRORE nella valutazione: {e}"
                                    st.error(f"Errore durante la valutazione di idoneità per '{file_name}': {e}")

                                st.markdown(f"#### 📄 {file_name}")
                                
                                if df_summary is not None and not df_summary.empty:
                                    bando_data_row = df_summary[df_summary["File"] == file_name]
                                    if not bando_data_row.empty:
                                        sintesi_data = bando_data_row.iloc[0].to_dict()
                                        st.markdown(f"**Titolo Avviso:** {sintesi_data.get('Titolo dell\'avviso', 'N/A')}")
                                        st.markdown(f"**Ente Erogatore:** {sintesi_data.get('Ente erogatore', 'N/A')}")
                                        st.markdown(f"**Sintesi Bando:** {sintesi_data.get('Descrizione aggiuntiva', 'N/A')}")
                                        if sintesi_data.get('Link') and sintesi_data.get('Link') != 'N/A':
                                            st.markdown(f"**Link:** [{sintesi_data.get('Link')}]({sintesi_data.get('Link')})")
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

    # --- TAB: HANDBOOK (RIASSUNTI) ---
    with tab_handbook:
        st.subheader("📖 Genera Handbook dei Bandi")
        st.info("Questa sezione crea un riassunto narrativo dettagliato per ciascun bando caricato, ideale per una consultazione rapida. I riassunti generati vengono memorizzati per velocizzare le consultazioni successive.")
        
        if active_chat["uploaded_file_names"]:
            selected_file_for_summary = st.selectbox(
                "Seleziona un file per generare il riassunto:",
                options=[""] + sorted(list(active_chat["uploaded_file_names"])),
                key=f"select_summary_file_{active_chat['id']}"
            )
            
            if selected_file_for_summary:
                # Controlla se il riassunto è già in cache
                if selected_file_for_summary in active_chat["summaries_cache"]:
                    st.info(f"Riassunto per '{selected_file_for_summary}' recuperato dalla cache.")
                    st.markdown(f"#### Riassunto per '{selected_file_for_summary}'")
                    st.write(active_chat["summaries_cache"][selected_file_for_summary])
                
                # Bottone per generare (o rigenerare) il riassunto
                if st.button(f"Genera Riassunto per '{selected_file_for_summary}'", key=f"generate_summary_btn_{active_chat['id']}"):
                    with st.spinner(f"Generazione del riassunto per '{selected_file_for_summary}'..."):
                        # Passiamo active_chat alla funzione extract_summary_for_file
                        summary_content = extract_summary_for_file(active_chat["pipeline"], selected_file_for_summary, active_chat)
                        st.markdown(f"#### Riassunto per '{selected_file_for_summary}'")
                        st.write(summary_content)
                        
                        # Opzione per scaricare il riassunto
                        st.download_button(
                            label="📥 Scarica Riassunto",
                            data=summary_content.encode("utf-8"),
                            file_name=f"riassunto_{selected_file_for_summary.replace('.', '_')}.txt",
                            mime="text/plain",
                            key=f"download_summary_{active_chat['id']}"
                        )
            else:
                st.info("Seleziona un file per generare il suo riassunto.")
        else:
            st.warning("Carica dei documenti nella sidebar per generare i riassunti.")



if __name__ == "__main__":
    main()




    