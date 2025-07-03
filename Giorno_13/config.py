"""
Configurazioni per NER Graph Extractor
"""

# Modello NER da utilizzare
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"

# Soglia confidence per accettare entità
MIN_CONFIDENCE = 0.5

# Tipi di entità da estrarre
ENTITY_TYPES = {
    "PER": "PERSON",        # Persone
    "ORG": "ORGANIZATION",  # Organizzazioni
    "LOC": "LOCATION",      # Luoghi
    "MISC": "MISC"          # Varie
}

# Pattern per riconoscere tipi specifici
PATTERNS = {
    "DATE": [
        r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',  # 15/03/2024
        r'\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}'     # 2024/03/15
    ],
    "MONEY": [
        r'€\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})?',     # €1.500,00
        r'\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*€',     # 1.500,00 €
        r'\d+[.,]\d{2}\s*euro?'                     # 1500.00 euro
    ],
    "EMAIL": [
        r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b'           # email@domain.com
    ]
}

# Colori per visualizzazione grafo
ENTITY_COLORS = {
    'PERSON': '#FF6B6B',        # Rosso
    'ORGANIZATION': '#4ECDC4',   # Turchese
    'LOCATION': '#45B7D1',       # Blu
    'DATE': '#96CEB4',           # Verde
    'MONEY': '#FFEAA7',          # Giallo
    'EMAIL': '#DDA0DD',          # Viola
    'OTHER': '#95A5A6'           # Grigio
}

# Configurazioni Streamlit
APP_CONFIG = {
    "page_title": "NER Graph Extractor",
    "page_icon": "🕸️",
    "layout": "wide"
}

# Limiti per performance
LIMITS = {
    "max_file_size_mb": 5,      # Massimo 5MB per file
    "max_files": 10,            # Massimo 10 file
    "max_nodes_display": 50     # Massimo 50 nodi nel grafo
}