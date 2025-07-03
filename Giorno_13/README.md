# 🕸️ NER Graph Extractor

**Estrai entità da documenti aziendali e costruisci grafi di conoscenza interattivi**

## 🎯 Che Cosa Fa

Questo progetto **standalone** ti permette di:

1. **📤 Caricare documenti** di testo (.txt)
2. **🔍 Estrarre automaticamente** entità (persone, aziende, date, importi)
3. **🔗 Rilevare relazioni** tra le entità
4. **🕸️ Visualizzare grafi interattivi** delle connessioni
5. **💾 Esportare risultati** in JSON e CSV

## 🚀 Quick Start

### Installazione
```bash
# 1. Clona o scarica il progetto
git clone <repository-url>
cd ner_graph_extractor

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Avvia l'app
streamlit run main.py
```

### Primo Test
1. Crea i documenti di esempio (vedi `sample_documents/`)
2. Carica i file nell'app
3. Clicca "Avvia Estrazione" 
4. Esplora il grafo interattivo!

## 📁 Struttura Progetto

```
ner_graph_extractor/
├── main.py              # App Streamlit principale  
├── ner_extractor.py     # Estrazione entità NER
├── graph_builder.py     # Costruzione grafi
├── config.py            # Configurazioni
├── requirements.txt     # Dipendenze
├── README.md           # Questa documentazione
└── sample_documents/   # Documenti di esempio
```

## 🎪 Funzionalità Principali

### 🔍 Estrazione Entità
- **PERSON**: Mario Rossi, Giulia Bianchi
- **ORGANIZATION**: ACME SpA, TechSolutions SRL  
- **LOCATION**: Milano, Via Roma 123
- **DATE**: 15/03/2024, 01/01/2024
- **MONEY**: €1.500,00, €2.300,00
- **EMAIL**: mario@email.com

### 🔗 Rilevamento Relazioni
- **LAVORA_PER**: Mario Rossi → ACME SpA
- **PAGAMENTO**: Cliente → €1.500,00 → Fornitore
- **UBICAZIONE**: ACME SpA → Milano
- **TEMPORALE**: Contratto → 01/01/2024
- **COMUNICAZIONE**: Persona → Email

### 🕸️ Visualizzazione Grafo
- **Grafo interattivo** con Plotly
- **Colori per tipo** entità
- **Hover dettagli** su nodi
- **Layout automatico** ottimizzato

### 💾 Export Risultati
- **JSON**: Grafo completo strutturato
- **CSV**: Tabelle entità e