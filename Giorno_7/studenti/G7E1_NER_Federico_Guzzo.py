from transformers import pipeline

# Inizializza il pipeline per il Named Entity Recognition (NER)
ner_model = "Davlan/bert-base-multilingual-cased-ner-hrl"
ner = pipeline("ner", model=ner_model, aggregation_strategy="simple")

# Testo di input
text = "Mario Rossi ha ricevuto un bonifico sull'IBAN IT60X0542811101000000123456."

# Estrae le entità dal testo
entities = ner(text)

# Stampa le entità riconosciute
print("Entità riconosciute:")
for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']}")
