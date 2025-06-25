# ---------------------------------------------------------------------------
# 1. IMPORT
# ---------------------------------------------------------------------------
import os
import glob
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# ---------------------------------------------------------------------------
# 2. CARICAMENTO MODELLO
# ---------------------------------------------------------------------------
print("🚀 Avvio anonimizzazione (dslim/bert-base-NER)")
try:
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    print("✅ Modello caricato!\n")
except Exception as e:
    print(f"❌ Errore nel caricamento del modello: {e}")
    raise SystemExit

# ---------------------------------------------------------------------------
# 3. FUNZIONE DI ANONIMIZZAZIONE
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "PER":  "[NOME_PERSONA]",
    "LOC":  "[INDIRIZZO]",
    "ORG":  "[NOME_ORGANIZZAZIONE]",
    "MISC": "[DATO_GENERICO]"
}

def anonymize(text: str) -> str:
    """Sostituisce le entità riconosciute con placeholder."""
    ents = sorted(ner(text), key=lambda x: len(x["word"]), reverse=True)
    for e in ents:
        text = text.replace(e["word"], LABEL_MAP.get(e["entity_group"], "[DATO_SENSIBILE]"))
    return text

# ---------------------------------------------------------------------------
# 4. PROCESSO FILE
# ---------------------------------------------------------------------------
def process_here():
    """
    Cerca ricorsivamente i .txt a partire dalla cartella dove risiede lo script
    e salva le versioni anonime in ./output_bert-base-NER/ accanto a ogni file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"🔍 Cartella di partenza: {base_dir}")

    pattern = os.path.join(base_dir, "**", "*.txt")
    all_txt = [p for p in glob.glob(pattern, recursive=True)
               if not os.path.basename(p).startswith("anon_")]

    if not all_txt:
        print("❌ Nessun .txt da anonimizzare.")
        return

    print(f"📄 Trovati {len(all_txt)} file:")
    for p in all_txt:
        print("  •", os.path.relpath(p, base_dir))

    for path in all_txt:
        try:
            with open(path, encoding="utf-8") as f:
                original = f.read()
            anon = anonymize(original)

            out_dir = os.path.join(os.path.dirname(path), "output_bert-base-NER")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"anon_{os.path.basename(path)}")

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(anon)

            print(f"✅ Salvato → {os.path.relpath(out_path, base_dir)}")
        except Exception as e:
            print(f"❌ Errore su {path}: {e}")

# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    process_here()
    print("\n--- ✅ Fine processo ---")
