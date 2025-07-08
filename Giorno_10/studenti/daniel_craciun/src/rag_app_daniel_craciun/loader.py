from pathlib import Path
from typing import Callable, Optional, List, Dict

class GenericFileLoader:
    """Loads files from a folder using a user-provided filter and reader."""

    def __init__(
        self,
        folder_path: str,
        file_filter: Optional[Callable[[Path], bool]] = None,
        file_reader: Optional[Callable[[Path], Dict]] = None,
    ) -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Path not found: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")
        self.file_filter = file_filter or (lambda f: f.suffix == ".txt")
        self.file_reader = file_reader or (
            lambda f: {"file_name": f.name, "content": f.read_text(encoding="utf-8")}
        )

    def load(self) -> List[Dict]:
        results = []
        for file in self.folder_path.iterdir():
            if self.file_filter(file):
                try:
                    results.append(self.file_reader(file))
                except Exception as e:
                    print(f"Error reading {file.name}: {e}")
        return results
