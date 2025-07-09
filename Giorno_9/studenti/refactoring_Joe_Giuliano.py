from transformers import pipeline

# Inizializza la pipeline di Named Entity Recognition (NER) con un modello multilingue
ner_pipe = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"
)

def estrai_entita(testuale):
    """
    Estrae e stampa le entità riconosciute in una stringa di testo.

    Parameters
    ----------
    testuale : str
        Il testo da analizzare.

    Returns
    -------
    list of dict
        Lista di entità riconosciute, ciascuna come dizionario.
    """
    entities = ner_pipe(testuale)
    for ent in entities:
        # Stampa il testo dell'entità e il suo tipo (es: "Leonardo: PER")
        print(f"{ent['word']}: {ent['entity_group']}")
    return entities

if __name__ == "__main__":
    # Test di esempio
    testo = "Leonardo da Vinci ha dipinto la Gioconda a Firenze."
    estrai_entita(testo)