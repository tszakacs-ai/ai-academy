# Giorno 14 - RAG con controllo di fairness

Questa directory contiene la versione rifattorizzata del progetto del giorno 8.
La struttura segue uno schema classico per progetti Python:


Il punto di ingresso dell'applicazione è `main.py`. Rispetto alla versione del
giorno 9 viene introdotto un controllo automatico sulle risposte generate dal
modello GPT. Dopo la generazione il testo è analizzato da un *BiasChecker*
basato su `transformers` che rileva eventuali espressioni tossiche o
discriminatorie. In caso di rischio viene registrato un log e l'utente riceve un
avviso.

## Configurazione
Le variabili sensibili vanno definite in un file `.env` posto nella stessa cartella. Un esempio di contenuto:

```
PROJECT_ENDPOINT=https://your-azure-endpoint
DEFAULT_FOLDER_PATH=./Dataset
```
`PROJECT_ENDPOINT` è obbligatoria per connettersi ad Azure AI Project, mentre `DEFAULT_FOLDER_PATH` indica la cartella da cui caricare i documenti di default.

Avviata l'applicazione, è possibile caricare uno o più file `.txt` dalla sidebar usando il widget di upload o modificare il percorso della cartella se necessario.

