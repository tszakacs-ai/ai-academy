# SMS Spam Classifier - Documentazione e Risk Assessment

## Descrizione
Questo progetto implementa un classificatore di SMS spam utilizzando un modello Naive Bayes. Il dataset utilizzato è il "SMS Spam Collection" di Kaggle.

## Dati
- Origine: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- Pulizia: Rimozione colonne inutili, mappatura label.

## Modello
- Algoritmo: Multinomial Naive Bayes
- Feature extraction: CountVectorizer

## Metriche
- Precisione, recall, confusion matrix

## Limiti e rischi
- Possibili falsi positivi/negativi
- Bias nei dati
- Necessità di aggiornamento periodico

## Analisi rischi e limiti secondo EU AI Act

Il modello SMS Spam Classifier, pur essendo “semplice”, ricade nelle categorie di rischio definite dalla normativa europea sull’AI. I principali rischi identificati sono:

- **Falsi positivi:** Possibile blocco di messaggi legittimi.
- **Falsi negativi:** Spam non rilevato, rischio sicurezza.
- **Bias:** Possibile discriminazione di certi utenti o contenuti.

## Obblighi di trasparenza e audit

- Documentazione dettagliata del modello e dei dati.
- Log delle predizioni e delle decisioni.
- Audit periodici e report di performance.
- Informazione agli utenti sull’uso di sistemi automatizzati.

## Documentazione e test richiesti

- Manuale tecnico e utente.
- Test di validazione, robustezza, bias.
- Audit log e report periodici.
- Aggiornamento continuo della documentazione e dei dati.

## Risk assessment
- Il modello ricade in una categoria di rischio "basso" secondo l’EU AI Act, ma richiede comunque documentazione, trasparenza e audit.