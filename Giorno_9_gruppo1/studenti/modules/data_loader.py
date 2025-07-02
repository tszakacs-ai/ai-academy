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
    
    # Verifica che il path esista e sia una cartella
    if not folder_path.exists():
        raise FileNotFoundError(f"Il path '{path}' non esiste")
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Il path '{path}' non è una cartella")
    
    documents = []
    
    # Cicla attraverso tutti i file nella cartella
    for file_path in folder_path.iterdir():
        # Salta le sottocartelle, processa solo i file
        if file_path.is_file():
            try:
                # Legge il contenuto del file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Crea il dizionario per questo file
                document = {
                    'title': file_path.name,  # Nome del file (senza path)
                    'content': content
                }
                
                documents.append(document)
            except Exception as e:
                print(f"[ERROR] Errore nella lettura del file '{file_path}': {e}")
    return documents