import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from config import *

class NERAnonymizer:
    def __init__(self, model_path: str):
        # NER model initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

        # Regex patterns for sensitive data
        self.patterns = [
            (re.compile(r"\bIT\d{2}[A-Z]\d{10}[A-Z0-9]{12}\b"), "[IBAN]"),
        ]

    def extract_entities(self, text: str) -> list[dict]:
        return self.ner_pipeline(text)

    def anonymize_text(self, text: str) -> str:
        # Step 1: NER
        ents = self.extract_entities(text)
        ents_sorted = sorted(ents, key=lambda e: e["end"], reverse=True)
        for ent in ents_sorted:
            tag = f"[{ent['entity_group']}]"
            if tag == "[PER]":
                text = text[:ent["start"]] + "[NAME]" + text[ent["end"]:]

        # Step 2: Regex
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)

        return text

    def anonymize_txt_file(self, input_path: str, output_path: str):
        with open(input_path, "r", encoding="utf-8") as infile:
            text = infile.read()

        print(f"Document read ({len(text)} characters)")

        anon_text = self.anonymize_text(text)

        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(anon_text)

        print(f"Anonymized document saved: {output_path}")


if __name__ == "__main__":

    model_path = MODEL_PATH
    try:
        anonymizer = NERAnonymizer(model_path)

        print(anonymizer.extract_entities("Sample text with Mario Rossi and IT60X0542811101000000123456"))
        anon_text = anonymizer.anonymize_text("Sample text with Mario Rossi and IT60X0542811101000000123456")
        print(anon_text)

        input_path = r"C:\Users\BG726XR\OneDrive - EY\Desktop\academy_profice\ai-academy-1\Giorno_8\Fattura.txt"
        output_path = r"C:\Users\BG726XR\OneDrive - EY\Desktop\academy_profice\ai-academy-1\Giorno_8\FatturaAnon.txt"

        anonymizer.anonymize_txt_file(input_path, output_path)
    except Exception as e:
        print(f"Error during initialization or execution: {e}")
