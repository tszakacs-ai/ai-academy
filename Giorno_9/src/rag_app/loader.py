from pathlib import Path

class TextFileLoader:
    """Carica tutti i file .txt da una cartella."""

    def __init__(self, folder_path: str) -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Percorso non trovato: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Non è una directory: {folder_path}")

    def load(self) -> list[dict]:
        results = []
        for file in self.folder_path.glob("*.txt"):
            try:
                content = file.read_text(encoding="utf-8")
                results.append({"file_name": file.name, "content": content})
            except Exception as e:  # pragma: no cover - logging
                print(f"Errore nella lettura di {file.name}: {e}")
        return results
