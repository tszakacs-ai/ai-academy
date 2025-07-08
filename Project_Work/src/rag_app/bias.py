from __future__ import annotations

from pathlib import Path
from transformers import pipeline


class BiasChecker:
    """Valuta un testo per rilevare potenziali bias o espressioni tossiche."""

    def __init__(self, threshold: float = 0.6, log_path: str = "bias_log.txt") -> None:
        self.threshold = threshold
        self.log_path = Path(log_path)
        try:
            self.classifier = pipeline("text-classification", model="unitary/toxic-bert")
        except Exception as e:  # pragma: no cover - handle missing model
            print(f"Errore caricamento modello bias: {e}")
            self.classifier = None

    def is_biased(self, text: str) -> bool:
        if not self.classifier:
            return False
        try:
            result = self.classifier(text)[0]
            label = result.get("label", "").lower()
            score = float(result.get("score", 0))
            if any(key in label for key in ["toxic", "hate", "offensive"]) and score >= self.threshold:
                self.log_bias(text, label, score)
                return True
        except Exception as e:  # pragma: no cover - log error
            print(f"Errore valutazione bias: {e}")
        return False

    def log_bias(self, text: str, label: str, score: float) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{label} {score:.2f}] {text}\n")
