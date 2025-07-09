import re
import networkx as nx
import pickle

def estrai_entita(testo):
    aziende = re.findall(r'\b[A-Z][a-zA-Z]+\sSrl\b', testo)
    date = re.findall(r'\d{1,2} [a-z]+ \d{4}', testo)
    importi = re.findall(r'\d{1,6} euro', testo)
    num_fattura = re.findall(r'n\.\d+', testo)
    prodotti = re.findall(r'prodotto\s\w+', testo, re.IGNORECASE)
    return {
        "Aziende": list(set(aziende)),
        "Date": list(set(date)),
        "Importi": list(set(importi)),
        "NumeroFattura": list(set(num_fattura)),
        "Prodotti": list(set(prodotti))
    }

def enrich_graph(input_path, output_path):
    # Usa pickle per compatibilità con tutte le versioni di networkx
    with open(input_path, "rb") as f:
        G = pickle.load(f)
    # Esempio: aggiungi un nodo "Auditor" collegato a tutti i documenti
    auditor = "Auditor Mario Rossi"
    G.add_node(auditor, tipo="Persona")
    for node, data in G.nodes(data=True):
        if data.get("tipo") == "Documento":
            G.add_edge(auditor, node, relazione="verifica")
    with open(output_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Grafo arricchito salvato in {output_path}")

if __name__ == "__main__":
    enrich_graph("../data/graph.gpickle", "../data/graph_enriched.gpickle")