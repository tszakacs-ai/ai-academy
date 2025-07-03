import networkx as nx
from transformers import pipeline
from pathlib import Path
import re
import matplotlib.pyplot as plt
import openai
import os
from neo4j import GraphDatabase
from py2neo import Graph

openai.api_key = os.getenv("OPENAI_API_KEY")
 
# Inizializza pipeline NER
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
 
# Inizializza grafo
G = nx.DiGraph()
doc_folder = Path("Giorno_10/data")
 
for file_path in doc_folder.glob("*"):
    if file_path.is_file():
        with open(file_path, 'r', encoding='utf-8') as f:
            testo = f.read()
 
        # Inizializza dizionario entità
        entita = {
            'persone': [],
            'aziende': [],
            'date': [],
            'importi': [],
            'prodotti': [],
            'numeri_fattura': [],
            'luoghi': [],
            'iban': [],
            'codici_fiscali': []
        }
 
        # NER
        entities = ner_pipeline(testo)
        for ent in entities:
            label = ent['entity_group']
            text_ent = ent.get('word', '').strip()
            if not text_ent:
                continue
 
            if label == 'PER':
                entita['persone'].append(text_ent)
            elif label == 'ORG':
                entita['aziende'].append(text_ent)
            elif label == 'DATE':
                entita['date'].append(text_ent)
            elif label == 'MISC' and re.search(r'\b\d{1,3}(?:\.\d{3})*,\d{2}\b|€', text_ent):
                entita['importi'].append(text_ent)
            elif label in ['LOC', 'GPE']:
                entita['luoghi'].append(text_ent)
 
        # Regex supplementari
        iban_pattern = r'[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}[A-Z0-9]{0,16}'
        cf_pattern = r'[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]'
        fattura_pattern = r'[nN]\.?\s?\d+'
 
        entita['iban'].extend(re.findall(iban_pattern, testo))
        entita['codici_fiscali'].extend(re.findall(cf_pattern, testo))
        entita['numeri_fattura'].extend(re.findall(fattura_pattern, testo))
 
        # Estrazione euristica prodotti/servizi
        prodotti_match = re.findall(r"(?:per|riguardante|relativa a|oggetto:)\s+([\w\s\-]{3,40})(?=\.|,|\n|$)", testo, re.IGNORECASE)
        entita['prodotti'].extend(prodotti_match)
 
        # Nodo documento
        doc_node = f"Doc_{file_path.stem}"
        G.add_node(doc_node, tipo="Documento")
 
        # Tipo documento inferito
        if "fattura" in testo.lower():
            tipo_doc = "Fattura"
        elif "contratto" in testo.lower():
            tipo_doc = "Contratto"
        elif "ordine" in testo.lower():
            tipo_doc = "Ordine"
        elif "email" in testo.lower() or "@gmail.com" in testo:
            tipo_doc = "Email"
        else:
            tipo_doc = "Generico"
        G.nodes[doc_node]['tipo_documento'] = tipo_doc
 
        # Emittente e destinatario
        aziende = list(dict.fromkeys(entita['aziende']))
        if len(aziende) >= 2:
            emittente, destinatario = aziende[0], aziende[1]
            G.add_node(emittente, tipo="Azienda")
            G.add_edge(doc_node, emittente, relazione="emessa_da")
            G.add_node(destinatario, tipo="Azienda")
            G.add_edge(doc_node, destinatario, relazione="inviata_a")
        else:
            for azienda in aziende:
                G.add_node(azienda, tipo="Azienda")
                G.add_edge(doc_node, azienda, relazione="coinvolge")
 
        for persona in set(entita['persone']):
            G.add_node(persona, tipo="Persona")
            G.add_edge(doc_node, persona, relazione="contiene")
 
        for data in set(entita['date']):
            G.add_node(data, tipo="Data")
            G.add_edge(doc_node, data, relazione="data_documento")
 
        for importo in set(entita['importi']):
            G.add_node(importo, tipo="Importo")
            G.add_edge(doc_node, importo, relazione="importo_totale")
 
        for num in set(entita['numeri_fattura']):
            G.add_node(num, tipo="NumeroFattura")
            G.add_edge(doc_node, num, relazione="identificata_da")
 
        for iban in set(entita['iban']):
            G.add_node(iban, tipo="IBAN")
            G.add_edge(doc_node, iban, relazione="pagabile_su")
 
        for cf in set(entita['codici_fiscali']):
            G.add_node(cf, tipo="CodFiscale")
            G.add_edge(doc_node, cf, relazione="cf_riferito")
 
        for prod in set(entita['prodotti']):
            G.add_node(prod.strip(), tipo="Prodotto")
            G.add_edge(doc_node, prod.strip(), relazione="oggetto")
 
        for luogo in set(entita['luoghi']):
            G.add_node(luogo, tipo="Luogo")
            G.add_edge(doc_node, luogo, relazione="localizzato_in")
 
# Pulizia dei nodi "vuoti" o stopword
stopwords = {'di', 'in', 'per', 'a', 'il', 'la', 'e'}
nodi_da_rimuovere = [n for n in G.nodes() if isinstance(n, str) and (len(n.strip()) < 2 or n.lower() in stopwords)]
for nodo in nodi_da_rimuovere:
    G.remove_node(nodo)
 
# Info generali
print(f"\nGrafo creato con {G.number_of_nodes()} nodi e {G.number_of_edges()} archi")
 
print("\nTipi di nodi:")
tipi_nodi = {}
for nodo, attr in G.nodes(data=True):
    tipo = attr.get('tipo', 'Sconosciuto')
    tipi_nodi[tipo] = tipi_nodi.get(tipo, 0) + 1
for tipo, count in tipi_nodi.items():
    print(f"- {tipo}: {count}")
 
 
 
# Stampa relazioni da un documento
documento = "Doc_fattura1"  
if G.has_node(documento):
    print(f"\nRelazioni da: {documento}")
    tipo_doc = G.nodes[documento].get("tipo_documento", "N/A")
    print(f"Tipo documento: {tipo_doc}")
    for succ in G.successors(documento):
        relazione = G[documento][succ].get('relazione', 'relazione_sconosciuta')
        tipo = G.nodes[succ].get('tipo', 'TipoSconosciuto')
        print(f"- {relazione}: {succ} ({tipo})")
else:
    print(f"Documento '{documento}' non trovato.")
 
# Visualizzazione
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, k=1.5, iterations=100)
labels = {n: (n[:12] + "...") if len(n) > 15 else n for n in G.nodes()}
 
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
 
edge_labels = nx.get_edge_attributes(G, 'relazione')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)
 
plt.title("Grafo Entità-Relazioni", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()

# --- Integrazione esercizio 2 ---

create_cliente_fattura_query = '''
CREATE (c:Cliente {nome: "Mario Rossi"})
CREATE (f:Fattura {numero: "INV2025"})
CREATE (c)-[:HA_RICEVUTO]->(f)
'''

match_fattura_query = '''
MATCH (f:Fattura)
WHERE f.importo > 1000
RETURN f.numero, f.importo
'''

domanda = "Quali clienti hanno ricevuto una fattura con importo superiore a 1000 euro?"

cypher_query = '''
MATCH (c:Cliente)-[:HA_RICEVUTO]->(f:Fattura)
WHERE f.importo > 1000
RETURN c.nome AS cliente, f.numero AS numero_fattura, f.importo AS importo
'''

risposta_query = [
    {"cliente": "Mario Rossi", "numero_fattura": "INV2025", "importo": 1500},
    {"cliente": "Luca Bianchi", "numero_fattura": "INV2026", "importo": 2000},
]


def riformula_risposta(risposta_query, domanda):
    tabella = "\n".join(
        [
            f"- {r['cliente']} ha ricevuto la fattura {r['numero_fattura']} di importo {r['importo']} euro"
            for r in risposta_query
        ]
    )
    prompt = (
        f"Domanda business: {domanda}\n"
        f"Risposta tabellare:\n{tabella}\n"
        "Riformula la risposta in linguaggio naturale chiaro e sintetico per un manager."
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.3,
    )
    return completion.choices[0].message.content.strip()


risposta_naturale = riformula_risposta(risposta_query, domanda)
print(risposta_naturale)

# --- Esempio di connessione a Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Password123")

# Utilizzo del driver ufficiale neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session() as session:
    result = session.run("MATCH (n) RETURN n LIMIT 5")
    for record in result:
        print(record)

# Utilizzo di py2neo
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
results = graph.run("MATCH (p:Persona) RETURN p.nome LIMIT 5")
for record in results:
    print(record)

