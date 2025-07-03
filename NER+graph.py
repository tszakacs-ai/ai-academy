from transformers import pipeline
import re
import networkx as nx
import matplotlib.pyplot as plt

ner_pipe = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", aggregation_strategy="simple")

text = (
    "La fattura n.123 è stata emezza dalla azienda Ciao Srl a Mario Rossi il 15/10/2023 per un importo di $1000. "
    "La fattura è stata pagata tramite bonifico bancario."
)
entities = ner_pipe(text)

# Regex for dates (dd/mm/yyyy or dd-mm-yyyy)
date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
dates = [{"word": m.group(), "entity_group": "DATE", "score": 1.0} for m in re.finditer(date_pattern, text)]

# Regex for currencies (€, $, £ followed by numbers)
currency_pattern = r"[\$\€\£]\s?\d+(?:[\.,]\d+)?"
currencies = [{"word": m.group(), "entity_group": "CURRENCY", "score": 1.0} for m in re.finditer(currency_pattern, text)]

# Regex for invoice number (fattura n.123)
fattura_pattern = r"fattura\s*n\.?\s*\d+"
fattura_match = re.search(fattura_pattern, text, re.IGNORECASE)
fattura_node = fattura_match.group().strip() if fattura_match else "Fattura"

# Combine all entities
entities.extend(dates)
entities.extend(currencies)

for entity in entities:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.2f}")

# Extraction of entities
orgs = [e for e in entities if e['entity_group'] == 'ORG']
pers = [e for e in entities if e['entity_group'] == 'PER']
locs = [e for e in entities if e['entity_group'] == 'LOC']
dates = [e for e in entities if e['entity_group'] == 'DATE']
currencies = [e for e in entities if e['entity_group'] == 'CURRENCY']

G = nx.DiGraph()

# Add central node for the invoice
G.add_node(fattura_node, tipo='FATTURA')

# Add nodes for all entity types and connect them to the fattura node
for org in orgs:
    G.add_node(org['word'], tipo='ORG')
    G.add_edge(fattura_node, org['word'], relazione='Emessa da')
for per in pers:
    G.add_node(per['word'], tipo='PER')
    G.add_edge(fattura_node, per['word'], relazione='Intestata a')
for loc in locs:
    G.add_node(loc['word'], tipo='LOC')
    G.add_edge(fattura_node, loc['word'], relazione='Luogo')
for date in dates:
    G.add_node(date['word'], tipo='DATE')
    G.add_edge(fattura_node, date['word'], relazione='Data')
for curr in currencies:
    G.add_node(curr['word'], tipo='CURRENCY')
    G.add_edge(fattura_node, curr['word'], relazione='Importo')

# Draw the graph
pos = nx.spring_layout(G)
node_labels = {n: f"{n}" for n in G.nodes}
nx.draw(G, pos, labels=node_labels, node_color='lightblue', node_size=700, font_size=12, font_color='black', edge_color='gray')

edge_labels = nx.get_edge_attributes(G, 'relazione')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

plt.title("Fattura-Centric NER + Regex Graph")
plt.show()