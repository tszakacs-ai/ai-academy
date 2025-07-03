import networkx as nx

G = nx.DiGraph()

# Nodi
G.add_node("n.123", tipo="NumeroFattura")
G.add_node("Alfa Srl", tipo="Fornitore")
G.add_node("Beta Srl", tipo="Cliente")
G.add_node("5 giugno 2024", tipo="Data")
G.add_node("2000 Euro", tipo="Importo")

# Relazioni
G.add_edge("n.123", "Alfa Srl", tipo="emesso da")
G.add_edge("n.123", "Beta Srl", tipo="inviato a")
G.add_edge("n.123", "5 giugno 2024", tipo="pagato il")
G.add_edge("n.123", "2000 Euro", tipo="importo")

# Visualizzazione del grafo
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_color='black', font_weight='bold', arrows=True)
edge_labels = nx.get_edge_attributes(G, 'tipo')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Rappresentazione della Fattura n.123")
plt.show()