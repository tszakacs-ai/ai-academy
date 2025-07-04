# Esercizio: Metriche di Modelli NLP

## NER + GPT: Metriche di Estrazione e Generazione

### Obiettivo
Pianificare la valutazione dell'accuratezza di estrazione entità e qualità delle risposte/generazione.

### Metriche Proposte

#### Per il modello NER
- **Precision**: Percentuale di entità identificate correttamente rispetto al totale delle entità estratte
- **Recall**: Percentuale di entità realmente presenti che sono state identificate dal modello
- **F1-score**: Media armonica di precision e recall, fornisce un bilanciamento tra i due valori

#### Per il modello GPT
- **BLEU**: Confronta n-grammi tra risposta generata e riferimento, utile per traduzioni e riscritture
- **ROUGE**: Misura la sovrapposizione lessicale, particolarmente efficace per riassunti
- **BERTScore**: Utilizza embedding semantici per valutare similarità di significato oltre la corrispondenza lessicale

### Metodologia di Valutazione

#### Scelta delle Metriche
Per NER utilizzerei F1-score come metrica principale, integrando precision e recall per analisi dettagliate per tipo di entità. Per GPT, la scelta dipende dal task: BLEU per generazione strutturata, ROUGE per riassunti, BERTScore per valutazioni semantiche.

#### Confronto degli Input
Preparerei dataset annotati manualmente come gold standard. Per NER confronterei le entità estratte con le annotazioni ground truth. Per GPT utilizzerei risposte di riferimento create da esperti umani.

#### Preparazione dei Dati
Creerei set di test bilanciati con esempi rappresentativi del dominio applicativo. Per NER includerebbe vari tipi di entità e contesti. Per GPT coprirebbe diversi stili di domande e complessità di risposta.

#### Calcolo delle Metriche
Per NER calcolerei precision, recall e F1 sia a livello globale che per categoria di entità. Per GPT applicerei le metriche scelte confrontando sistematicamente output del modello con riferimenti gold standard, considerando anche valutazioni umane per aspetti qualitativi non catturati dalle metriche automatiche.