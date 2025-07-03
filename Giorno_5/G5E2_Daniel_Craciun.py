import json
from transformers import pipeline
 
# NER per identificare dati da mascherare
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
 
# Lettura contenuto mail
with open("Giorno_5/mail_daniel.txt", "r", encoding="utf-8") as f:
    text = f.read()
 
# Mascheramento dei dati sensibili usando NER
entities = ner(text)
masked_text = text
for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
    label = ent["entity_group"]
    start, end = ent["start"], ent["end"]
    replacement = {
        "PER": "[NOME]",
        "EMAIL": "[EMAIL]",
        "PHONE": "[TELEFONO]",
    }.get(label, None)
    if replacement:
        masked_text = masked_text[:start] + replacement + masked_text[end:]
 
# Prompt da inviare a TinyLlama (modello tuo originale)
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
prompt = f"""
Estrai i seguenti dati dal testo qui sotto:
- nome completo
- indirizzo email
- numero di telefono
 
Restituisci solo un JSON con le chiavi: "nome", "email", "telefono".
 
Testo:
{masked_text}
"""
 
# Generazione con TinyLlama
raw_output = pipe(prompt, max_new_tokens=256)[0]['generated_text']
 
# Parsing del JSON
try:
    json_start = raw_output.find('{')
    json_end = raw_output.rfind('}') + 1
    json_string = raw_output[json_start:json_end]
    output = json.loads(json_string)
    print("Nome:", output.get("nome"))
    print("Email:", output.get("email"))
    print("Telefono:", output.get("telefono"))
except Exception as e:
    print("Errore durante il parsing JSON:", e)
    print("Contenuto grezzo generato:", raw_output)