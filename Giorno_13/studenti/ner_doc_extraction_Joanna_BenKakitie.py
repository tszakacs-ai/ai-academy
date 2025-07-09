from transformers import pipeline
import re
import networkx as nx

# Inizializza la pipeline NER
ner_pipe = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", aggregation_strategy="simple")

# Unisci i testi in una singola stringa
text = (
    "Mario Rossi ha ricevuto un bonifico sull'IBAN IT60X054281101000000123456 "
    "La fattura n.123 emessa da Alfa Srl a Beta Srl il 5 giugno 2024 per un importo di 2000 Euro."
)
entities = ner_pipe(text)

# Regex per le date
date_pattern = r"\b\d{1,2} [a-zA-Z]+ \d{4}\b"
dates = [{"word": m.group(0), "entity_group": "DATE", "score":1.0} for m in re.finditer(date_pattern, text)]

# Regex per le correnti
currency_pattern = r"\b\d+(\.\d{1,2})?\s*(Euro|euro|EUR)\b"
currencies = [{"word": m.group(0), "entity_group": "CURRENCY", "score":1.0} for m in re.finditer(currency_pattern, text)]

# Regex per i numeri di fattura
invoice_pattern = r"fattura\s? n\.\d+"
invoice_match = re.search(invoice_pattern, text, re.IGNORECASE)
invoice_node = invoice_match.group().strip() if invoice_match else "Fattura"

# Conbinazione delle entità NER con le regex
entities.extend(dates)
entities.extend(currencies)

for entity in entities:
    print(f"Entità: {entity['word']}, Tipo: {entity['entity_group']}, Punteggio: {entity['score']}")

# Estrazione delle entità
orgs = [e for e in entities if e['entity_group'] == 'ORG']
pers = [e for e in entities if e['entity_group'] == 'PER']
locs = [e for e in entities if e['entity_group'] == 'LOC']
dates = [e for e in entities if e['entity_group'] == 'DATE']
currencies = [e for e in entities if e['entity_group'] == 'CURRENCY']

# Creazione del grafo
G = nx.DiGraph()

# Nodi e archi per le entità
G.add_node(invoice_node, tipo="Fattura") # Nodo centrale

for org in orgs:
    G.add_node(org['word'], tipo="ORG")
    G.add_edge(invoice_node, org['word'], tipo="emesso da")
for per in pers:
    G.add_node(per['word'], tipo="PER")
    G.add_edge(invoice_node, per['word'], tipo="emesso a")
for loc in locs:
    G.add_node(loc['word'], tipo="LOC")
    G.add_edge(invoice_node, loc['word'], tipo="località")
for date in dates:
    G.add_node(date['word'], tipo="Data")
    G.add_edge(invoice_node, date['word'], tipo="data")
for currency in currencies:
    G.add_node(currency['word'], tipo="Importo")
    G.add_edge(invoice_node, currency['word'], tipo="importo")


# Visualizzazione del grafo
import matplotlib.pyplot as plt

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_color='black', font_weight='bold', arrows=True)
edge_labels = nx.get_edge_attributes(G, 'tipo')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Rappresentazione della Fattura n.123")
plt.show()