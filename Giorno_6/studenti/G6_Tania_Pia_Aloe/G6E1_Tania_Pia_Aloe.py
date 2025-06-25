import os
from openai import AzureOpenAI

# --- CONFIGURAZIONE AZURE OPENAI ---
AZURE_ENDPOINT = ""
AZURE_KEY = ""
AZURE_DEPLOYMENT = "gpt-4.1-mini"  
API_VERSION = "2024-12-01-preview"

# Cartella con i file di testo dei documenti
DOCS_DIR = "Giorno_6/studenti/G6_Tania_Pia_Aloe/File_txt"

# Prompt base per classificare il tipo di documento
SYSTEM_PROMPT = """
Sei un assistente AI che deve identificare il tipo di un documento fornito.

Rispondi SOLO con una delle seguenti categorie:
'Email', 'Nota di Credito', 'Fattura', 'Contratto', 'Sconosciuto'

Non aggiungere altro testo o spiegazioni.
"""

def classify_document(file_path: str) -> str:
    """
    Legge il contenuto del file, costruisce il prompt e chiama Azure OpenAI
    per classificare il tipo di documento.
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version=API_VERSION
    )

    with open(file_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    user_prompt = (
        "Determina la categoria aziendale del seguente documento, "
        "basandoti SOLO sul testo fornito.\n\n"
        "Documento:\n"
        f"{doc_text}\n\n"
        "Rispondi solo con la categoria."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        temperature=0.0,   # meno creatività, più precisione
        max_tokens=50
    )

    answer = response.choices[0].message.content.strip()
    return answer

def main():
    if not os.path.exists(DOCS_DIR):
        print(f"❌ Cartella '{DOCS_DIR}' non trovata.")
        return

    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    if not files:
        print(f"❌ Nessun file .txt trovato in '{DOCS_DIR}'.")
        return

    for file_name in files:
        file_path = os.path.join(DOCS_DIR, file_name)
        print(f"\nAnalisi file: {file_name}")
        categoria = classify_document(file_path)
        print(f"Categoria rilevata: {categoria}")

if __name__ == "__main__":
    main()