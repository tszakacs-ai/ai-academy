# 🤖 Sistema RAG con Anonimizzazione dei Dati

Un sistema avanzato di Retrieval-Augmented Generation (RAG) che combina l'anonimizzazione automatica dei documenti con un chatbot intelligente basato su Azure OpenAI. Il sistema è progettato per processare documenti sensibili, anonimizzarli automaticamente e permettere interrogazioni sicure attraverso un'interfaccia conversazionale.

## ✨ Caratteristiche Principali

### 🔒 Anonimizzazione Intelligente
- **NER (Named Entity Recognition)** per identificare e anonimizzare nomi di persone
- **Pattern matching** per IBAN e altri dati sensibili
- Utilizzo del modello BERT italiano specializzato (`osiria/bert-italian-cased-ner`)

### 🧠 Sistema RAG Avanzato
- **Ricerca semantica** con embeddings Azure OpenAI
- **Chunking intelligente** dei documenti con overlap per mantenere il contesto
- **Indicizzazione vettoriale** con FAISS per ricerche ultra-rapide
- **Memoria conversazionale** per mantenere il contesto tra le domande

### 💻 Interfacce Multiple
- **Interfaccia Terminale**: Per uso diretto da command line
- **Interfaccia Web Streamlit**: Dashboard interattiva e user-friendly
- **Gestione documenti**: Upload, visualizzazione e gestione dei file

### 📊 Funzionalità Avanzate
- **Citazione delle fonti** con punteggi di rilevanza
- **Salvataggio conversazioni** in formato JSON
- **Riassunti automatici** delle conversazioni
- **Gestione memoria** con limite di token e messaggi

## 🚀 Installazione

### Prerequisiti
- Python 3.8+
- Account Azure OpenAI con deployments attivi
- Git

### Setup del Progetto

1. **Clona il repository**
```bash
git clone <url-repository>
cd rag-anonymizer-system
```

2. **Installa le dipendenze**
```bash
pip install -r requirements.txt
```

3. **Configurazione delle credenziali**

Crea un file `.env` nella root del progetto:
```env
AZURE_ENDPOINT_RAG=https://your-resource.openai.azure.com/
API_KEY_RAG=your-api-key-here
EMBEDDING_DEPLOYMENT_RAG=text-embedding-ada-002
GPT_DEPLOYMENT_RAG=gpt-4
```

4. **Configurazione dei percorsi**

Modifica il file `config.py` con i tuoi percorsi:
```python
MODEL_PATH = "path/to/your/model"
FOLDER_PATH = "path/to/input/documents"
ANON_FILE_PATH = "path/to/anonymized/documents"
RESULTS_PATH = "path/to/results"
HISTORY_PATH = "path/to/chat/history.json"
```

## 📖 Utilizzo

### Interfaccia Streamlit (Raccomandato)

Avvia l'interfaccia web interattiva:
```bash
python main.py streamlit
```

**Workflow nell'interfaccia web:**
1. **Upload documenti**: Carica i tuoi file TXT nella sidebar
2. **Inizializza RAG**: Clicca "Inizializza RAG" per processare i documenti
3. **Chat interattiva**: Inizia a fare domande sui tuoi documenti
4. **Gestisci conversazioni**: Salva, cancella o riassumi le tue chat

### Interfaccia Terminale

Per un'esperienza da command line:
```bash
python main.py terminal
```

**Comandi disponibili:**
- `quit` - Chiude la chat
- `clear` - Cancella la storia della conversazione
- `summary` - Mostra un riassunto della conversazione
- `save` - Salva la conversazione in un file JSON

### Solo Anonimizzazione

Per processare solo l'anonimizzazione senza avviare la chat:
```bash
python main.py
```

## 🏗️ Architettura del Sistema

### Componenti Principali

#### 🔍 NERAnonymizer (`ner_anonymizer.py`)
- Gestisce l'anonimizzazione dei documenti
- Utilizza BERT per NER in italiano
- Pattern regex per dati strutturati (IBAN, ecc.)

#### 🤖 AzureRAGSystem (`AzureRag.py`)
- Core del sistema RAG
- Gestione embeddings e ricerca vettoriale
- Memoria conversazionale avanzata
- Integrazione con Azure OpenAI

#### 📁 DataLoader (`data_loader.py`)
- Caricamento e gestione dei documenti
- Supporto per cartelle con multipli file
- Gestione errori robusta

#### 🖥️ Interfacce
- **main.py**: Orchestratore principale e interfaccia terminale
- **streamlit_app.py**: Dashboard web con Streamlit

### Flusso di Elaborazione

```
Documenti Originali → Anonimizzazione → Chunking → Embeddings → Indicizzazione → RAG
                                                                                    ↓
                                                                            Chat Interface
```

## 🔧 Configurazione Avanzata

### Parametri di Chunking
```python
chunk_size = 1000  # Caratteri per chunk
overlap = 200      # Overlap tra chunk
```

### Memoria Conversazionale
```python
max_history_messages = 20  # Massimo 20 messaggi
max_history_tokens = 4000  # Token massimi per la storia
```

### Ricerca Semantica
```python
top_k = 3           # Documenti da recuperare
max_tokens = 2000   # Token massimi per risposta
```

## 📊 Monitoraggio e Debugging

### Logs del Sistema
Il sistema fornisce logging dettagliato per:
- Processo di anonimizzazione
- Caricamento documenti
- Generazione embeddings
- Ricerche e risposte

### Metriche Disponibili
- Numero di documenti processati
- Chunks indicizzati
- Scambi conversazionali
- Token utilizzati per richiesta

## 🛡️ Sicurezza e Privacy

### Dati Anonimizzati
- **Nomi di persone** → `[NAME]`
- **IBAN** → `[IBAN]`
- **Estensibile** per altri pattern

### Best Practices
- I documenti originali vengono mantenuti separati
- I documenti anonimizzati sono salvati in cartelle dedicate
- Nessun dato sensibile viene inviato ad Azure senza anonimizzazione

## 🔄 Estensibilità

### Aggiungere Nuovi Pattern di Anonimizzazione
```python
# In ner_anonymizer.py
self.patterns.append(
    (re.compile(r"pattern_regex"), "[REPLACEMENT]")
)
```

### Personalizzare il Sistema RAG
```python
# Modificare parametri in AzureRag.py
self.max_history_messages = 30  # Più memoria
chunk_size = 1500              # Chunks più grandi
```

## 📋 TODO e Miglioramenti Futuri

- [ ] Supporto per più formati di file (PDF, DOCX)
- [ ] Anonimizzazione di indirizzi email e numeri di telefono
- [ ] Interfaccia REST API
- [ ] Deploy con Docker
- [ ] Testing automatizzato
- [ ] Supporto multilingua

## 🤝 Contribuire

1. Fork del progetto
2. Crea un branch per la tua feature (`git checkout -b feature/nuova-feature`)
3. Commit delle modifiche (`git commit -am 'Aggiunge nuova feature'`)
4. Push del branch (`git push origin feature/nuova-feature`)
5. Apri una Pull Request

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per i dettagli.

## 📞 Supporto

Per domande, bug report o richieste di feature, apri un issue su GitHub.

---

**Nota**: Assicurati di configurare correttamente le credenziali Azure e i percorsi prima dell'utilizzo. Il sistema è ottimizzato per documenti in italiano ma può essere adattato per altre lingue.