from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
import os

nlp= pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", aggregation_strategy="simple")
print("Pipeline NER caricata con successo.")

def censura_testo_ner(testo) :
    """
    Esegue la Named Entity Recognition su un testo e censura le entità sensibili
    sostituendole con etichette corrispondenti.
    """
    if not testo:
        return ""

    # Esegui la NER sul testo
    risultati_ner = nlp(testo)
    print("\nRisultati NER raw:")
    for res in risultati_ner:
        print(res)

    testo_censurato = list(testo) # Converti il testo in una lista di caratteri per facilitare la sostituzione

    # Ordina i risultati NER in ordine decrescente di offset per evitare problemi con gli indici
    risultati_ner_ordinati = sorted(risultati_ner, key=lambda x: x['start'], reverse=True)

    for entita in risultati_ner_ordinati:
        start = entita['start']
        end = entita['end']
        label = entita['entity_group'] # Utilizziamo entity_group per etichette aggregate (es. PER, LOC, ORG)
        word = entita['word']

        sostituzione = f'[{label}]' # Se non c'è una mappatura specifica, usa l'etichetta originale
        testo_censurato[start:end] = list(sostituzione) #Sostituzione

    return "".join(testo_censurato)


# Per l'IBAN, il modello non è addestrato specificamente per riconoscerlo.
# è necessaria una regex per rilevare e censurare gli IBAN.
def censura_iban(testo) :
    iban_regex = r'\bIT\d{2}[A-Z0-9]{1,23}\b'
    return re.sub(iban_regex, '[IBAN]', testo)


# esempio
testo_esempio = (
    "Il Dott. Marco Verdi ha un appuntamento il 22/07/2025 alle 10:00 "
    "presso l'Ospedale Civile di Milano, in Piazza Duomo 1. "
    "Il suo numero di telefono è +39 333 1234567 e la sua email marco.verdi@email.com. "
    "L'IBAN per il pagamento è IT98X0123456789012345678901. "
    "Ciao piacere sono Alessandro"
)

print(f"\nTesto Originale:\n{testo_esempio}")
testo_censurato_passo1 = censura_testo_ner(testo_esempio)
testo_censurato_finale = censura_iban(testo_censurato_passo1)
print(f"\nTesto Censurato Finale:\n{testo_censurato_finale}")