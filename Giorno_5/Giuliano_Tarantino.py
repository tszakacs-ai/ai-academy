import os
import json
import re
import ssl
from typing import Union
from transformers import pipeline
from dotenv import load_dotenv
import openai


# SSL workaround (solo se necessario per certificati personalizzati)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["REQUESTS_CA_BUNDLE"] = r"C:\\Users\\SE645QY\\custom-ca-bundle.pem"


class NERExtractor:
    def __init__(self):
        # Inizializza il modello di Named Entity Recognition (NER)
        self.ner_pipe = pipeline(
            task="token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device_map="auto"
        )

    def extract_entities(self, text: str) -> dict:
        """
        Estrae entità nominate dal testo e le organizza in categorie.

        Parameters
        ----------
        text : str
            Il testo da analizzare.

        Returns
        -------
        dict
            Dizionario con le entità estratte organizzate per categoria.
        """
        result = {key: [] for key in ["PERSON", "ORG", "LOC", "DATE", "BANKCODE", "TAXCODE", "MISC"]}

        for ent in self.ner_pipe(text):
            tag = ent["entity_group"]
            word = ent["word"].strip()

            if tag in result:
                result[tag].append(word)
            else:
                result["MISC"].append(word)

        # Rimuove duplicati ignorando maiuscole/minuscole
        for key in result:
            seen = set()
            result[key] = [x for x in result[key] if not (x.lower() in seen or seen.add(x.lower()))]

        return result


class EmailAnonymizer:
    def __init__(self, input_folder: str = "input_documents", output_folder: str = "output_documents") -> None:
        """
        Inizializza l'oggetto EmailAnonymizer.

        Parameters
        ----------
        input_folder : str
            Cartella contenente i file di input.
        output_folder : str
            Cartella dove salvare i file anonimizzati.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Crea la cartella di output se non esiste
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output folder ready: {self.output_folder}")

        # Carica le variabili d'ambiente per Azure OpenAI
        load_dotenv()
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        azure_api_key = os.getenv("AZURE_API_KEY")

        self.client = openai.AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-12-01-preview",
        )

        self.extractor = NERExtractor()

    def log_debug(self, label: str, content: Union[str, dict, list]) -> None:
        """
        Logga informazioni di debug in modo leggibile.

        Parameters
        ----------
        label : str
            Etichetta per il log.
        content : Union[str, dict, list]
            Contenuto da loggare.
        """
        print(f"\n[LOG] {label}\n{'-' * 60}")
        if isinstance(content, (dict, list)):
            print(json.dumps(content, indent=2, ensure_ascii=False))
        else:
            print(content)

    def anonymize_email(self, text: str, entities: dict) -> str:
        """
        Anonimizza il testo sostituendo le entità con segnaposto.

        Parameters
        ----------
        text : str
            Il testo da anonimizzare.
        entities : dict
            Dizionario con le entità estratte.

        Returns
        -------
        str
            Testo anonimizzato.
        """
        redacted_text = text
        for label, items in entities.items():
            for item in items:
                escaped_value = re.escape(item)
                placeholder = f"[{label}]"
                redacted_text = re.sub(escaped_value, placeholder, redacted_text)
        return redacted_text

    def process_file(self, filename: str) -> None:
        """
        Processa un singolo file per anonimizzarlo.

        Parameters
        ----------
        filename : str
            Nome del file da processare.
        """
        filepath = os.path.join(self.input_folder, filename)

        with open(filepath, "r", encoding="utf-8") as fp:
            text = fp.read()
        self.log_debug("Original Text (truncated)", text[:500] + ("..." if len(text) > 500 else ""))

        entities = self.extractor.extract_entities(text)
        self.log_debug("Detected Entities", entities)

        if not any(entities.values()):
            print("[INFO] No entities found. Skipping file.")
            return

        anonymized_email = self.anonymize_email(text, entities)
        self.log_debug("Locally Anonymized Email (truncated)", anonymized_email[:500] + ("..." if len(anonymized_email) > 500 else ""))

        prompt = f"""
        You are a privacy assistant helping to redact sensitive information from documents.
        Below is a document that has already been partially anonymized.  
        Use the list of named entities provided to ensure all sensitive data has been fully replaced with appropriate placeholders (e.g., [PERSON], [IBAN], [ORG], etc.).
        Do not attempt to reconstruct or guess any original information.
        --- Entities extracted ---
        {entities}
        --- Anonymized Document ---
        {anonymized_email}
        Please return only the fully anonymized version of the document.
        """
        self.log_debug("Prompt Sent to GPT-4o (truncated)", prompt[:500] + "...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data-privacy assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=2024,
                temperature=0.3
            )

            redacted_text = response.choices[0].message.content
            self.log_debug("GPT-4o Response (truncated)", redacted_text[:500] + ("..." if len(redacted_text) > 500 else ""))

            output_filename = f"{os.path.splitext(filename)[0]}_anonymized.txt"
            output_path = os.path.join(self.output_folder, output_filename)

            with open(output_path, "w", encoding="utf-8") as out_fp:
                out_fp.write(redacted_text)

            print(f"[SUCCESS] File saved to: {output_path}")

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

    def main(self) -> None:
        """
        Processa tutti i file nella cartella di input.
        """
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".txt"):
                print(f"\n[INFO] Processing file: {filename}")
                self.process_file(filename)


if __name__ == "__main__":
    anonymizer = EmailAnonymizer()
    anonymizer.main()