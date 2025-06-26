<<<<<<< HEAD
from transformers import pipeline

ner_pipe = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"
)

text = "Mario Rossi ha ricevuto un bonifico sull’IBAN IT60X0542811101000000123456."
entities = ner_pipe(text)

# Stampa risultato: Mario Rossi
entities = ner_pipe(text)
for ent in entities:
    print(f"{ent['word']}: {ent['entity_group']}")
=======
from transformers import pipeline

ner_pipe = pipeline(
    "ner", 
    model = "Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy = "simple"
)

text = "Mario Rossi ha ricevuto un bonifico sull'IBAN IT60X054281101000000123456."
entities = ner_pipe(text)

# Stampa il risultato: Mario Rossi
entities = ner_pipe(text)
for ent in entities:
    print(f"{ent['word']}: {ent['entity_group']}")
>>>>>>> features/joanna-benkakitie
