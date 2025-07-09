from pathlib import Path

# --- CONFIGURAZIONE GLOBALE DELL'APPLICAZIONE ---
APP_TITLE = "Analisi e Estrazione Bandi AI (RAG AI)"

TMP_UPLOADS_PATH = Path("tmp_uploads")
TMP_UPLOADS_PATH.mkdir(exist_ok=True)

SAVED_CHATS_FOLDER = "saved_chats"
Path(SAVED_CHATS_FOLDER).mkdir(exist_ok=True)

# --- DOMANDE PER L'ESTRAZIONE STRUTTURATA (Tabella Excel) ---
TEMPLATE_FIELDS = [
    "Ente erogatore",
    "Titolo dell'avviso",
    "Descrizione aggiuntiva",
    "Beneficiari",
    "Apertura",
    "Chiusura",
    "Dotazione finanziaria",
    "Contributo",
    "Note",
    "Link",
    "Key Words",
    "Aperto (si/no)",
]

TEMPLATE_QUESTIONS = {
    "Ente erogatore": (
        "Scrivi solo il nome esatto dell’ente erogatore di questo bando, "
        "scegliendolo dalle prime tre pagine. Se non lo trovi, deduci quello "
        "più probabile dal testo. Solo il nome, nessuna spiegazione."
    ),
    "Titolo dell'avviso": (
        "Scrivi solo il titolo ufficiale dell’avviso, così come appare o "
        "come puoi dedurlo dalle prime tre pagine. Solo la dicitura, nessuna "
        "spiegazione."
    ),
    "Descrizione aggiuntiva": (
        "Scrivi una sola frase molto breve (massimo 25 parole) che riassume "
        "l’intero bando. Solo la frase, senza spiegazioni."
    ),
    "Beneficiari": (
        "Scrivi solo i beneficiari principali di questo bando, anche dedotti "
        "dal testo. Solo l’elenco, senza spiegazioni."
    ),
    "Apertura": (
        "Scrivi solo la data di apertura (formato GG/MM/AAAA), anche dedotta "
        "dal testo se non è esplicitata."
    ),
    "Chiusura": (
        "Scrivi solo la data di chiusura (formato GG/MM/AAAA), anche dedotta "
        "dal testo se non è esplicitata."
    ),
    "Dotazione finanziaria": (
        "Qual è la dotazione finanziaria totale del bando? Scrivi solo la "
        "cifra o il valore principale della dotazione finanziaria, anche se "
        "devi dedurlo dal testo."
    ),
    "Contributo": (
        "Qual è il contributo previsto per i beneficiari? Scrivi solo la cifra "
        "o percentuale principale del contributo previsto, anche se la deduci "
        "dal testo."
    ),
    "Note": (
        "Scrivi solo una nota rilevante, anche se la deduci dal testo. Solo "
        "la nota, senza spiegazioni."
    ),
    "Link": (
        "Scrivi solo il link (URL) principale trovato nel testo, oppure "
        "deducilo se presente in altro modo."
    ),
    "Key Words": (
        "Scrivi solo tre parole chiave, anche dedotte dal testo, separate da "
        "virgola e senza spiegazioni."
    ),
    "Aperto (si/no)": (
        "Rispondi solo con 'si' o 'no' se il bando è ancora aperto; deduci la "
        "risposta dal testo e dalle date. Nessuna spiegazione."
    ),
}

__all__ = [
    "APP_TITLE",
    "TMP_UPLOADS_PATH",
    "SAVED_CHATS_FOLDER",
    "TEMPLATE_FIELDS",
    "TEMPLATE_QUESTIONS",
]
