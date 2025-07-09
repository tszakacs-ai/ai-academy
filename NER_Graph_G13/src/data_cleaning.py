import os
import pandas as pd
import networkx as nx
import pickle
from cryptography.fernet import Fernet

# Percorso alla cartella dei dati
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

def clean_entities(input_path, output_path):
    """
    Pulisce il file CSV rimuovendo spazi, duplicati e valori nulli.
    """
    df = pd.read_csv(input_path)

    # Rimuove spazi e converte tutto in stringa
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # Rimuove duplicati e righe con valori nulli
    df = df.drop_duplicates()
    df = df.dropna()

    # Salva il file pulito
    df.to_csv(output_path, index=False)
    print(f"File pulito salvato in {output_path}")

def crea_grafo_e_critta(output_path, key_path=os.path.join(DATA_DIR, 'graph.key')):
    """
    Crea un grafo dai dati puliti e lo salva in formato crittografato.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, 'entities_clean.csv'))
    G = nx.MultiDiGraph()

    for idx, row in df.iterrows():
        doc = row.get('Documento', f"Documento_{idx}")
        G.add_node(doc, tipo="Documento")

        if 'Azienda' in row and pd.notna(row['Azienda']):
            G.add_node(row['Azienda'], tipo="Azienda")
            G.add_edge(doc, row['Azienda'], relazione="contiene")

        if 'Data' in row and pd.notna(row['Data']):
            G.add_node(row['Data'], tipo="Data")
            G.add_edge(doc, row['Data'], relazione="data")

        if 'Importo' in row and pd.notna(row['Importo']):
            G.add_node(str(row['Importo']), tipo="Importo")
            G.add_edge(doc, str(row['Importo']), relazione="importo")

        if 'NumeroFattura' in row and pd.notna(row['NumeroFattura']):
            G.add_node(row['NumeroFattura'], tipo="NumeroFattura")
            G.add_edge(doc, row['NumeroFattura'], relazione="numero")

        if 'Prodotto' in row and pd.notna(row['Prodotto']):
            G.add_node(row['Prodotto'], tipo="Prodotto")
            G.add_edge(doc, row['Prodotto'], relazione="prodotto")

    # Serializza il grafo
    graph_bytes = pickle.dumps(G)

    # Assicura che la directory esista
    os.makedirs(os.path.dirname(key_path), exist_ok=True)

    # Genera o carica la chiave di cifratura
    if not os.path.exists(key_path):
        key = Fernet.generate_key()
        with open(key_path, 'wb') as kf:
            kf.write(key)
    else:
        with open(key_path, 'rb') as kf:
            key = kf.read()

    # Cifra il grafo
    f = Fernet(key)
    encrypted_graph = f.encrypt(graph_bytes)

    # Salva il grafo crittografato
    with open(output_path, 'wb') as ef:
        ef.write(encrypted_graph)

    print(f"Grafo crittografato salvato in {output_path}")

if __name__ == "__main__":
    input_csv = os.path.join(DATA_DIR, 'entities.csv')
    output_csv = os.path.join(DATA_DIR, 'entities_clean.csv')
    encrypted_graph_path = os.path.join(DATA_DIR, 'grafo_criptato.bin')

    clean_entities(input_csv, output_csv)
    crea_grafo_e_critta(encrypted_graph_path)
