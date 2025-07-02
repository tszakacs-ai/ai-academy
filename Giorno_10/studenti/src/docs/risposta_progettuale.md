- Obiettivo
Il problema consisteva nell’elaborare automaticamente documenti contenenti dati sensibili (es. fatture, email clienti) per:
- Anonimizzare le informazioni riservate.
- Estrarre informazioni semantiche rilevanti.
- Generare una risposta formale coerente e professionale.

- Soluzione proposta  
Abbiamo sviluppato una pipeline automatizzata in due fasi principali:
- Anonimizzazione con un modello NER locale (osiriabert) per la protezione della privacy.
- Analisi semantica e generazione di risposta usando un modello GPT-4o via Azure OpenAI, orchestrato tramite LangChain.

- Scelte progettuali chiave

| **Componente**           | **Scelta**                                           | **Motivazione**                                                                  |
| ------------------------ | ---------------------------------------------------- | --------------------------------------------------------------------------------  | 
| **Modello NER**          | `osiriabert-italian-cased-ner`                       | Specializzato per l’italiano, eseguito localmente, senza invio di dati sensibili |
| **Modello LLM**          | `GPT-4o` via Azure OpenAI                            | Prestazioni elevate, supporto multilingua, integrazione diretta con LangChain    |
| **Framework LLM**        | [LangChain](https://www.langchain.com/)              | Consente composizione modulare di catene LLM e facilità di orchestrazione        |
| **Tipologia prompt**     | Prompt combinato oppure modulari (3 prompt separati) | Permette flessibilità a seconda della complessità e del contesto del documento   |
| **Struttura del codice** | Classi `NERAnonymizer` e `ChatbotAnalyzer` separate  | Separazione delle responsabilità, riutilizzabilità e testabilità                 |
| **Persistenza entità**   | Mapping JSON `tag → valore`                          | Consente reinserimento controllato delle entità nelle risposte                   |
| **Estendibilità**        | Input da file `.txt` parametrico                     | Consente scalabilità a qualsiasi tipo di documento (fatture, email, report)      |
