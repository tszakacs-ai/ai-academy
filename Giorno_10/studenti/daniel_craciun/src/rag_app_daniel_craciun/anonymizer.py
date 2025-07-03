import re
from typing import Dict, Callable, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class TextAnonymizer:
    """
    Masks sensitive entities using NER and regex. Now fully configurable for patterns and masks.
    """
    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        regex_patterns: Optional[Dict[str, str]] = None,
        entity_mask_map: Optional[Dict[str, str]] = None,
        aggregation_strategy: str = "simple",
        custom_ner_pipeline: Optional[Callable] = None,
    ) -> None:
        if custom_ner_pipeline:
            self.ner_pipeline = custom_ner_pipeline
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner", model=model, tokenizer=tokenizer, aggregation_strategy=aggregation_strategy
            )

        # Default patterns and masks are empty, user can provide any
        self.regex_patterns = regex_patterns or {}
        self.entity_mask_map = entity_mask_map or {}

    def mask_text(self, text: str) -> str:
        masked_text = text
        # Regex masking
        for label, pattern in self.regex_patterns.items():
            matches = re.findall(pattern, masked_text)
            for match in matches:
                masked_text = masked_text.replace(
                    match, self.entity_mask_map.get(label, "[MASKED]")
                )
        # NER masking
        entities = self.ner_pipeline(masked_text)
        for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
            label = ent.get("entity_group") or ent.get("entity")
            if label in self.entity_mask_map:
                masked_text = (
                    masked_text[: ent["start"]]
                    + self.entity_mask_map[label]
                    + masked_text[ent["end"] :]
                )
        return masked_text
