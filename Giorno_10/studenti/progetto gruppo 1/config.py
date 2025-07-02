from pathlib import Path

parent_dir = Path(__file__).parent

# Project paths
MODEL_PATH = parent_dir / "osiriabert-italian-cased-ner"
FOLDER_PATH = parent_dir / "data" / "dataset"
ANON_FILE_PATH = parent_dir / "data" / "file_anon" 
RESULTS_PATH = parent_dir / "data" / "results"
ENV_PATH = parent_dir / ".env"
HISTORY_PATH = parent_dir / "history"