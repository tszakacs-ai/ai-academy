from pathlib import Path

from PyPDF2 import PdfReader


class PdfFileLoader:
    """Carica tutti i file .pdf da una cartella."""

    def __init__(self, folder_path: str) -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Percorso non trovato: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Non è una directory: {folder_path}")

    def load(self) -> list[dict]:
        results = []
        for file in self.folder_path.glob("*.pdf"):
            try:
                reader = PdfReader(str(file))
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
                results.append({"file_name": file.name, "content": content})
            except Exception as e:  # pragma: no cover - logging
                print(f"Errore nella lettura di {file.name}: {e}")
        return results
