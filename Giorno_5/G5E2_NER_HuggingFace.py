from transformers import pipeline

# Inizializza la pipeline di Named Entity Recognition (NER)
ner_pipe = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"
)

# Testo da analizzare
text = "Mario Rossi ha ricevuto un bonifico sull'IBAN IT60X0542811101000000123456."

# Estrae le entità dal testo
entities = ner_pipe(text)

# Stampa le entità riconosciute
for ent in entities:
    print(f"{ent['word']}: {ent['entity_group']}")
