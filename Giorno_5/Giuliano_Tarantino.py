import json
import re
from transformers import pipeline
import spacy

# 1. Inizializzazione del modello NER (Named Entity Recognition) con Hugging Face
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# 2. Caricamento del modello SpaCy per tokenizzazione (opzionale)
nlp_spacy = spacy.load("en_core_web_sm")

# 3. Lettura del file di testo contenente le email
with open("Giorno_5/Mail.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 4. Prima passata: utilizzo del modello NER per identificare entità e mascherarle
entities = ner(text)
masked_text = text
for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
    label = ent["entity_group"]  # Tipo di entità (es. PER, EMAIL, PHONE)
    start, end = ent["start"], ent["end"]  # Posizioni dell'entità nel testo
    replacement = {
        "PER": "[NOME]",  # Maschera per i nomi di persone
        "EMAIL": "[EMAIL]",  # Maschera per le email
        "PHONE": "[TEL]",  # Maschera per i numeri di telefono
    }.get(label, None)
    if replacement:
        masked_text = masked_text[:start] + replacement + masked_text[end:]

# 5. Seconda passata: utilizzo di un modello di generazione testo per estrarre informazioni
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
prompt = (
    "Estrai nome, email e numero di telefono dal testo:\n"
    f"{masked_text}\nRestituisci solo JSON con chiavi nome, email, telefono."
)
raw_output = pipe(prompt, max_new_tokens=256)[0]['generated_text']

# 6. Parsing sicuro del JSON generato dal modello
try:
    j = raw_output.find('{')  # Trova l'inizio del JSON
    k = raw_output.rfind('}') + 1  # Trova la fine del JSON
    output = json.loads(raw_output[j:k])  # Carica il JSON
    print("Nome:", output.get("nome"))  # Stampa il nome estratto
    print("Email:", output.get("email"))  # Stampa l'email estratta
    print("Telefono:", output.get("telefono"))  # Stampa il numero di telefono estratto
except Exception as e:
    print("Errore parsing:", e)  # Gestione degli errori di parsing
    print("Risposta completa:", raw_output)  # Mostra l'output completo per debug
