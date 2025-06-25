from transformers import pipeline

# Carica il modello NER multilingua
ner_pipe = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"
)

# Leggi il testo dal file
with open("./Giorno_6/Mail.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Trova le entità
entities = ner_pipe(text)

# Sostituisci le entità con placeholder
masked_text = text
for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
    label = ent["entity_group"]
    start, end = ent["start"], ent["end"]
    replacement = {
        "PER": "[NOME]",
        "ORG": "[AZIENDA]",
        "LOC": "[LUOGO]",
        "MISC": "[ALTRO]"
    }.get(label, f"[{label}]")
    masked_text = masked_text[:start] + replacement + masked_text[end:]

print("Testo anonimizzato:\n")
print(masked_text)