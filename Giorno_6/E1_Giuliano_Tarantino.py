from transformers import pipeline

# Inizializza il modello di generazione testo TinyLlama da Hugging Face
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Legge il contenuto del file "nota.txt" che contiene una nota di credito amministrativa
with open("Giorno_6/nota.txt", "r", encoding="utf-8") as f:
    testo_nota = f.read()

# Crea un prompt per generare 5 domande basate sul contenuto della nota di credito
# Le domande devono riguardare dati fiscali, dettagli della fornitura stornata,
# importo totale e condizioni del rimborso
prompt = f"""
Il seguente testo è una nota di credito amministrativa:

{testo_nota}

Genera 5 domande di verifica sul contenuto di questa nota di credito. Le domande devono riguardare i dati fiscali, i dettagli della fornitura stornata, l'importo totale e le condizioni del rimborso. Scrivi le domande in italiano in modo chiaro e preciso.
"""

# Genera le domande utilizzando il modello di generazione testo
output = pipe(prompt, max_new_tokens=300)[0]['generated_text']

# Stampa le domande generate dal modello
print("Domande generate:")
print(output)
