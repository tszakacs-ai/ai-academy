# API Reference

## RAG System

### `search_documents(clients, query, index_name, top_k=5)`

Cerca documenti rilevanti nel database vettoriale basandosi sulla semantica della query.

**Parametri:**
- `clients`: Oggetto contenente i client necessari (Azure OpenAI, Pinecone)
- `query`: La query di ricerca dell'utente
- `index_name`: Il nome dell'indice Pinecone da interrogare
- `top_k`: Numero massimo di risultati da restituire (default: 5)

**Ritorna:**
- Lista di oggetti `QueryResult` contenenti i documenti più rilevanti

### `generate_rag_response(clients, query, relevant_docs, model_deployment, chat_history=None)`

Genera una risposta basata sui documenti rilevanti trovati.

**Parametri:**
- `clients`: Oggetto contenente i client necessari
- `query`: La query dell'utente
- `relevant_docs`: Lista di documenti rilevanti trovati
- `model_deployment`: Nome del deployment del modello di chat
- `chat_history`: Storico opzionale della chat (default: None)

**Ritorna:**
- Tupla contenente (risposta_generata, lista_fonti)

## PDF Query System

### `process_pdf_query(query, clients, config, top_k=5)`

Processa una query contro documenti PDF utilizzando embeddings e ricerca vettoriale.

**Parametri:**
- `query`: La query dell'utente
- `clients`: I client per Azure OpenAI e Pinecone
- `config`: Configurazione dell'applicazione
- `top_k`: Numero di risultati da restituire (default: 5)

**Ritorna:**
- Lista di risultati rilevanti dal database vettoriale

## Email Analysis

### `extract_questions_from_email(email_text, client)`

Estrae domande o richieste da un'email utilizzando il modello di linguaggio.

**Parametri:**
- `email_text`: Il testo dell'email da analizzare
- `client`: Client per Azure OpenAI

**Ritorna:**
- Lista di domande estratte dall'email

### `analyze_email_metadata(email_text, client)`

Analizza l'email per estrarre metadati come tipo di cliente e livello di urgenza.

**Parametri:**
- `email_text`: Il testo dell'email da analizzare
- `client`: Client per Azure OpenAI

**Ritorna:**
- Dizionario con i metadati (customer_type, urgency_level)
