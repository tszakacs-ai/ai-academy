import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class TextAnonymizer:
    """Maschera entità sensibili usando NER e regex."""

    def __init__(self) -> None:
        model_name = "dslim/bert-base-NER"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
        )

        self.regex_patterns = {
            "IBAN": r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b",
            "CF": r"\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b",
            "PHONE": r"\b(\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{5,8}\b",
        }

        self.entity_mask_map = {
            "PER": "[NOME]",
            "LOC": "[INDIRIZZO]",
            "ORG": "[AZIENDA]",
            "MISC": "[VARIABILE]",
            "IBAN": "[IBAN]",
            "CF": "[CF]",
            "PHONE": "[TELEFONO]",
        }

    def mask_text(self, text: str) -> str:
        masked_text = text
        for label, pattern in self.regex_patterns.items():
            matches = re.findall(pattern, masked_text)
            for match in matches:
                masked_text = masked_text.replace(
                    match, self.entity_mask_map.get(label, "[SENSIBILE]")
                )
        entities = self.ner_pipeline(masked_text)
        for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
            label = ent["entity_group"]
            if label in self.entity_mask_map:
                masked_text = (
                    masked_text[: ent["start"]]
                    + self.entity_mask_map[label]
                    + masked_text[ent["end"] :]
                )
        return masked_text
