# Architettura del Sistema MOCA AI

## Panoramica

Questo documento descrive l'architettura del sistema Consulente Normativo MOCA AI, un'applicazione basata su RAG (Retrieval Augmented Generation) per la consultazione di normative sui Materiali e Oggetti a Contatto con Alimenti.

## Componenti Principali

![Architettura del Sistema](architectural_scheme_moca.png)

### 1. Interfaccia Utente (UI)
- Implementata con **Streamlit**
- Due modalità di interazione:
  - **Chat Normativa**: Per domande dirette sulla normativa
  - **Gestione Email**: Per analisi e risposta automatica alle email dei clienti

### 2. Moduli di Backend
- **RAG System**: Nucleo del sistema che gestisce la ricerca semantica e la generazione delle risposte
- **PDF Query System**: Sistema specializzato per l'interrogazione di documenti PDF
- **Email Analyzer**: Modulo per l'estrazione di domande e metadati dalle email

### 3. Servizi Cloud
- **Azure OpenAI**:
  - Modello di chat (GPT-4o) per la generazione delle risposte
  - Modello di embedding (Ada 002) per la rappresentazione vettoriale dei testi
- **Pinecone**: Database vettoriale per lo storage e la ricerca semantica dei documenti

## Flusso di Dati

1. **Preprocessing**:
   - Documenti PDF vengono suddivisi in chunk
   - I chunk vengono trasformati in embeddings
   - Gli embeddings vengono caricati in Pinecone

2. **Query Processing**:
   - L'utente invia una domanda tramite UI
   - La domanda viene trasformata in embedding
   - Si esegue una ricerca semantica in Pinecone
   - I documenti rilevanti vengono recuperati

3. **Risposta**:
   - I documenti rilevanti vengono inviati al modello GPT
   - Il modello genera una risposta contestualizzata
   - La risposta viene presentata all'utente con le fonti

## Struttura del Progetto

```
/
├── src/               # Codice sorgente principale
├── data/              # File di dati (embeddings, dataset)
├── tests/             # Test automatici
├── notebooks/         # Notebook Jupyter
├── docs/              # Documentazione
└── app.py             # Entry point dell'applicazione
```
