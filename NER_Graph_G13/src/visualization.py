import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(gpickle_path):
    G = nx.read_gpickle(gpickle_path)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'relazione')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Property Graph NER Documenti Aziendali")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_graph("../data/graph_enriched.gpickle")