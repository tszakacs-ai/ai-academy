# Giorno 9 - Refactoring RAG

Questa directory contiene la versione rifattorizzata del progetto del giorno 8.
La struttura segue uno schema classico per progetti Python:

- `src/` codice sorgente dell'applicazione
- `data/` file di test o dataset
- `tests/` script di test automatici
- `notebooks/` notebook Jupyter per esperimenti
- `docs/` documentazione aggiuntiva

Il punto di ingresso dell'applicazione è `main.py`.

## Configurazione
Le variabili sensibili vanno definite in un file `.env` posto nella stessa cartella. Un esempio di contenuto:

```
PROJECT_ENDPOINT=https://your-azure-endpoint
DEFAULT_FOLDER_PATH=./data
```

`PROJECT_ENDPOINT` è obbligatoria per connettersi ad Azure AI Project, mentre `DEFAULT_FOLDER_PATH` imposta la cartella predefinita da cui caricare i documenti.
