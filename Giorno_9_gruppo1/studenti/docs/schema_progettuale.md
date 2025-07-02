              +-----------------------+
              |      Documento        |
              |    originale .txt     |
              +----------+------------+
                         |
                         v
         +------------------------------+
         |  NERAnonymizer (modello NER) |
         | - Carica modello locale      |
         | - Rileva entità sensibili    |
         | - Anonimizza contenuto       |
         +--------------+---------------+
                         |
                         v
            +-------------------------+
            |  File anonimizzato.txt  |
            +------------+------------+
                         |
                         v
         +-------------------------------+
         | ChatbotAnalyzer (LangChain +  |
         | Azure OpenAI / GPT-4o)        |
         | - Riepilogo                   |
         | - Analisi semantica          |
         | - Risposta cliente            |
         +---------------+---------------+
                         |
                         v
         +-------------------------------+
         |     Output formattato         |
         | (testo analizzato e risposta) |
         +-------------------------------+