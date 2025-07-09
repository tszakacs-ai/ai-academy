# Guida Utente

## Introduzione

Il Consulente Normativo MOCA AI è un assistente intelligente progettato per supportare gli utenti nella navigazione e comprensione delle normative relative ai Materiali e Oggetti a Contatto con Alimenti (MOCA). L'applicazione sfrutta l'intelligenza artificiale e la tecnologia RAG (Retrieval Augmented Generation) per fornire risposte accurate e contestualizzate.

## Installazione

### Prerequisiti
- Python 3.9 o superiore
- Account Azure con servizio OpenAI configurato
- Account Pinecone con un indice creato

### Passaggi
1. Clona il repository
2. Installa le dipendenze: `pip install -r requirements.txt`
3. Crea un file `.env` nella directory principale con le seguenti variabili:
   ```
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_DEPLOYMENT=your_model_deployment_name
   AZURE_EMBEDDING_DEPLOYMENT=your_embedding_model_name
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_index_name
   ```
4. Avvia l'applicazione: `streamlit run app.py`

## Utilizzo

### Modalità Chat Normativa

1. Seleziona "💬 Chat Normativa" dalla barra laterale
2. Digita la tua domanda nel campo di input in fondo alla pagina
3. Ricevi risposte basate sui documenti normativi con citazioni delle fonti
4. Puoi iniziare nuove conversazioni cliccando su "➕ Nuova conversazione"

### Modalità Gestione Email

1. Seleziona "📧 Gestione Email" dalla barra laterale
2. Seleziona un'email dall'elenco nella barra laterale
3. Visualizza l'analisi dell'email che include:
   - Tipo di cliente
   - Livello di urgenza
   - Domande/richieste identificate
4. Clicca su "🤖 Genera risposta automatica" per ottenere una bozza di risposta
5. Modifica la risposta se necessario e copiala negli appunti

## Consigli per domande efficaci

- Sii specifico nelle tue domande
- Menziona articoli o regolamenti specifici se li conosci
- Per argomenti complessi, dividi la domanda in parti più semplici
- Usa termini tecnici corretti per ottenere risposte più precise

## Limitazioni

- Le risposte sono generate sulla base dei documenti disponibili nell'indice
- L'applicazione non sostituisce la consulenza legale professionale
- Alcune normative molto recenti potrebbero non essere incluse nella base di conoscenza
