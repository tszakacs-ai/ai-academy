import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# === CONFIG ===
MODEL_PATH = r"C:\Users\QT153ZL\OneDrive - EY\Desktop\matthewthomson-ai-academy\Giorno_7\osiriabert-italian-cased-ner"

# Caricamento modello
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
_ner_pipeline = pipeline(
    "ner",
    model=_model,
    tokenizer=_tokenizer,
    aggregation_strategy="simple"
)

# === REGEX pattern ===
PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\+?\d[\d\-\s]{6,}\d"), "[PHONE]"),
    (re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b"), "[IBAN]"),
    (re.compile(r"\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b"), "[CF]"),
    (re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"), "[DATA]")
]

# === FUNZIONI ===
def estrai_entita(text: str) -> list[dict]:
    """
    Estrae entità NER dal testo.
    """
    return _ner_pipeline(text)

def anonimizza_testo(text: str) -> str:
    """
    Anonimizza il testo: NER + regex.
    """
    # --- Fase 1: NER ---
    ents = estrai_entita(text)
    ents_sorted = sorted(ents, key=lambda e: e["end"], reverse=True)
    anon_text = text
    for e in ents_sorted:
        tag = f"[{e['entity_group']}]"
        anon_text = anon_text[:e["start"]] + tag + anon_text[e["end"]:]

    # --- Fase 2: Regex ---
    for pattern, replacement in PATTERNS:
        anon_text = pattern.sub(replacement, anon_text)

    return anon_text

def anonimizza_file_txt(input_path: str, output_path: str):
    """
    Legge un file txt, anonimizza e scrive su output txt.
    """
    with open(input_path, "r", encoding="utf-8") as infile:
        testo = infile.read()
    
    print(f"Documento letto ({len(testo)} caratteri)")

    testo_anonimizzato = anonimizza_testo(testo)

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(testo_anonimizzato)
    
    print(f"Documento anonimizzato salvato: {output_path}")

# === Esempio ===
if __name__ == "__main__":
    input_file = "documento_test.txt"
    output_file = "documento_test_anonimizzato.txt"

    anonimizza_file_txt(input_file, output_file)
