from pathlib import Path
from typing import List, Dict

def load_data(path: Path) -> List[Dict[str, str]]:
    """
    Legge tutti i file in una cartella e restituisce una lista di dizionari
    con title (nome file) e content (contenuto file).
    
    Args:
        path (str): Percorso della cartella da leggere
        
    Returns:
        List[Dict[str, str]]: Lista di dizionari con 'title' e 'content'
        
    Raises:
        FileNotFoundError: Se il path non esiste
        NotADirectoryError: Se il path non è una cartella
    """
    folder_path = Path(path)
    
    # existing checks
    if not folder_path.exists():
        raise FileNotFoundError(f"Il path '{path}' non esiste")
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Il path '{path}' non è una cartella")
    
    documents = []
    
    # loop through all files in the folder
    for file_path in folder_path.iterdir():
        # avoid directories, only process files
        if file_path.is_file():
            try:
                # read the file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # create a document dictionary
                document = {
                    'title': file_path.name, 
                    'content': content
                }
                
                documents.append(document)
            except Exception as e:
                print(f"[ERROR] Errore nella lettura del file '{file_path}': {e}")
    return documents