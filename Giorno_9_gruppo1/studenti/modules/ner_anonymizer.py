import re
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
from config import *

class NERAnonymizer:
    def __init__(self, model_path: str):
        # NER model initialization
        self.tokenizer = BertTokenizerFast.from_pretrained("osiria/bert-italian-cased-ner")
        self.model = BertForTokenClassification.from_pretrained("osiria/bert-italian-cased-ner")
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
