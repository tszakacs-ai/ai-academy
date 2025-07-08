from transformers import pipeline

# Inizializza il generatore di testo con TinyLlama
generatore = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Caricamento della nota di credito dal file
percorso_file = "studenti/RAGDocs/nota_di_credito.txt"
with open(percorso_file, "r", encoding="utf-8") as file:
    contenuto_nota = file.read()

# Prompt per la generazione delle domande
prompt_domande = f"""\
Hai a disposizione il seguente documento, che rappresenta una nota di credito amministrativa:

{contenuto_nota}

Formula 5 domande per verificare la comprensione di questa nota. Le domande devono riguardare:
- le informazioni fiscali presenti,
- i dettagli della fornitura stornata,
- l’importo complessivo,
- e le condizioni relative al rimborso.

Scrivi le domande in italiano in modo chiaro, preciso e formale.
"""

# Generazione delle domande tramite il modello
risultato = generatore(prompt_domande, max_new_tokens=300)[0]["generated_text"]

# Output finale
print("📄 Domande generate dalla nota di credito:")
print(risultato)
