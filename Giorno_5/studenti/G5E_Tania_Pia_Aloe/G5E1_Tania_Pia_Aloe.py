# --- SETUP E IMPORT ---
import os
import glob
from openai import AzureOpenAI

print("\nEsecuzione script di riepilogo testi anonimizzati con Azure OpenAI")

# --- CONFIGURAZIONE AZURE OPENAI ---
AZURE_ENDPOINT = ""
AZURE_KEY = ""
AZURE_DEPLOYMENT = "gpt-4.1-mini"  
API_VERSION = "2024-12-01-preview"

# Cartella input (con i file anonimizzati)
INPUT_DIR_NAME = "output_bert-base-NER"
OUTPUT_DIR_NAME = "output_azure_summaries"

# ---------------------------------------------------------------------------
# 3. FUNZIONE DI ELABORAZIONE FILE CON AZURE OPENAI
# ---------------------------------------------------------------------------
def process_anonymized_texts():
    """
    Cerca ricorsivamente tutti i file 'anon_*.txt' dentro la cartella input,
    li elabora con Azure OpenAI (prompt per riassunto),
    e salva il risultato in output.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, INPUT_DIR_NAME)
    output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)

    print(f"🔍 Cartella base: {base_dir}")
    print(f"📂 Cartella input: {input_dir}")

    if not os.path.exists(input_dir):
        print(f"❌ Cartella input '{INPUT_DIR_NAME}' non trovata.")
        return

    # Cerca ricorsivamente i file anon_*.txt
    pattern = os.path.join(input_dir, "**", "anon_*.txt")
    file_list = glob.glob(pattern, recursive=True)

    if not file_list:
        print(f"❌ Nessun file 'anon_*.txt' trovato in '{INPUT_DIR_NAME}'.")
        return

    print(f"📄 Trovati {len(file_list)} file anonimizzati.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Cartella output creata o già esistente: {output_dir}")

    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version=API_VERSION
        )
        print("✅ Client Azure OpenAI configurato correttamente.")
    except Exception as e:
        print(f"❌ Errore nella configurazione del client Azure: {e}")
        return

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        print(f"\n🧠 Elaborazione file: {file_name}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                testo_anonimizzato = f.read()

            prompt = f"Riassumi il seguente testo anonimizzato in massimo 5 righe:\n\n{testo_anonimizzato}"

            messages = [{"role": "user", "content": prompt}]

            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )

            summary = response.choices[0].message.content

            output_path = os.path.join(output_dir, f"summary_{file_name}")
            with open(output_path, "w", encoding="utf-8") as f_out:
                f_out.write(summary)

            print(f"✅ Riassunto salvato in: {os.path.relpath(output_path, base_dir)}")

        except Exception as e:
            print(f"❌ Errore nell'elaborazione di {file_name}: {e}")

# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    process_anonymized_texts()
    print("\n--- ✅ Processo completato ---")