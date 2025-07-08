import re

class TextAnonymizer:
    """Maschera entità sensibili usando solo regex."""

    def __init__(self) -> None:

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
        return masked_text
