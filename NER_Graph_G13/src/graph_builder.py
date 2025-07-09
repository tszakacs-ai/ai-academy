import pandas as pd
import networkx as nx

def build_graph(input_path, output_path):
    df = pd.read_csv(input_path)
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
    nx.write_gpickle(G, "../data/graph.gpickle")
    print(f"Grafo salvato in ../data/graph.gpickle")

if __name__ == "__main__":
    build_graph("../data/entities_normalized.csv", "../data/graph.gpickle")