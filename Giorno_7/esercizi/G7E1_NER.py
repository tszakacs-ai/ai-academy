<<<<<<< HEAD
from transformers import pipeline

ner_pipe = pipeline(
    "ner",
    model="LR-AI-Labs/tiny-universal-NER",
    aggregation_strategy="simple"
)

text = "Mario Rossi ha ricevuto un bonifico sull’IBAN IT60X0542811101000000123456."
entities = ner_pipe(text)
print(entities)
=======
from transformers import pipeline

ner_pipe = pipeline(
    "ner", 
    model = "LR-AI-Labs/tiny-universal-NER",
    aggregation_strategy = "simple"
)

text = "Mario Rossi ha ricevuto un bonifico sull'IBAN IT60X054281101000000123456"
entities = ner_pipe(text)
print(entities)
>>>>>>> features/joanna-benkakitie
