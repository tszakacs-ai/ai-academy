"""
Configurazione per il sistema di anonimizzazione NER con grafi
"""

class Config:
    """Configurazioni principali del sistema"""
    
    # Modello NER da utilizzare
    NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"
    # Alternativamente puoi usare:
    # NER_MODEL = "dslim/bert-base-NER"
    # NER_MODEL = "Jean-Baptiste/camembert-ner"  # Per italiano
    
    # Soglia di confidenza per le entità NER
    NER_CONFIDENCE_THRESHOLD = 0.5
    
    # Impostazioni Streamlit
    ST_PAGE_TITLE = "Sistema di Anonimizzazione con NER e Grafi"
    ST_PAGE_ICON = "🔒"


# Pattern regex per diversi tipi di entità
REGEX_PATTERNS = {
    # Codice Fiscale italiano
    "CODICE_FISCALE": r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b',
    
    # Partita IVA italiana
    "PARTITA_IVA": r'\b\d{11}\b',
    
    # Email
    "EMAIL": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    
    # Numeri di telefono italiani
    "TELEFONO": r'\b(?:\+39\s?)?(?:0\d{1,4}[-.\s]?)?\d{6,8}\b',
    
    # Date in formato italiano (dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy)
    "DATA": r'\b(?:0[1-9]|[12][0-9]|3[01])[\/\-\.](0[1-9]|1[012])[\/\-\.](?:19|20)\d{2}\b',
    
    # Codice IBAN
    "IBAN": r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{0,16}\b',
    
    # Numeri di carta di credito
    "CARTA_CREDITO": r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
    
    # Indirizzi IP
    "IP_ADDRESS": r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
    
    # URL
    "URL": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
    
    # Numeri di documento (passaporto, patente, etc.)
    "NUMERO_DOCUMENTO": r'\b[A-Z]{2}\d{7}\b',
    
    # Importi in euro
    "IMPORTO": r'\b\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s?€|\b€\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?\b',
    
    # Indirizzi (Via, Piazza, etc.)
    "INDIRIZZO": r'\b(?:Via|Piazza|Viale|Corso|Largo|Vicolo)\s+[A-Za-z\s]+\d+\b',
    
    # CAP italiano
    "CAP": r'\b\d{5}\b',
    
    # Coordinate bancarie (ABI, CAB)
    "COORDINATE_BANCARIE": r'\b\d{5}[-.\s]?\d{5}\b',
}

# Mapping per tradurre le etichette NER in italiano
NER_LABELS_MAPPING = {
    'PER': 'PERSONA',
    'PERSON': 'PERSONA',
    'LOC': 'LUOGO',
    'LOCATION': 'LUOGO',
    'ORG': 'ORGANIZZAZIONE',
    'ORGANIZATION': 'ORGANIZZAZIONE',
    'MISC': 'VARIE',
    'MISCELLANEOUS': 'VARIE',
    'B-PER': 'PERSONA',
    'I-PER': 'PERSONA',
    'B-LOC': 'LUOGO',
    'I-LOC': 'LUOGO',
    'B-ORG': 'ORGANIZZAZIONE',
    'I-ORG': 'ORGANIZZAZIONE',
    'B-MISC': 'VARIE',
    'I-MISC': 'VARIE',
}

# Colori per la visualizzazione del grafo
GRAPH_COLORS = {
    'PERSONA': '#FF6B6B',
    'LUOGO': '#4ECDC4',
    'ORGANIZZAZIONE': '#45B7D1',
    'EMAIL': '#96CEB4',
    'TELEFONO': '#FFEAA7',
    'CODICE_FISCALE': '#DDA0DD',
    'PARTITA_IVA': '#98D8C8',
    'DATA': '#F7DC6F',
    'IBAN': '#BB8FCE',
    'CARTA_CREDITO': '#F1948A',
    'INDIRIZZO': '#85C1E9',
    'IMPORTO': '#82E0AA',
    'Documento': '#D5DBDB',
    'Categoria': '#AED6F1',
    'default': '#BDC3C7'
}

# Impostazioni per la visualizzazione del grafo
GRAPH_LAYOUT_CONFIG = {
    'spring': {
        'k': 3,
        'iterations': 50
    },
    'circular': {},
    'random': {},
    'kamada_kawai': {}
}