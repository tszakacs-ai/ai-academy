from transformers import pipeline


ner_pipe = pipeline(
    "ner",
    model = "Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"

)

text = "Mario Rossi ha ricevuto un bonifico sul iban IT60X0542811101000000123456 da Giovanni Bianchi per l'acquisto di un computer portatile."
entities = ner_pipe(text)

for ent in entities:
    print(f"Entità: {ent['word']}, Tipo: {ent['entity_group']}, Punteggio: {ent['score']:.2f}")