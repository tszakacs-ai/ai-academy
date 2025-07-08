import networkx as nx
from transformers import pipeline
from pathlib import Path
import re
import matplotlib.pyplot as plt
import difflib
from py2neo import Graph, Node, Relationship

 
# Inizializza pipeline NER
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
 
# Inizializza grafo
G = nx.DiGraph()
doc_folder = Path("./Giorno_13/NER_Graph_Esercizio/Docs")
 
def normalizza_nomi(nomi, soglia=0.8):
    gruppi = []
    codici = {}
    for nome in nomi:
        trovato = False
        for gruppo in gruppi:
            if difflib.SequenceMatcher(None, nome.lower(), gruppo[0].lower()).ratio() > soglia:
                gruppo.append(nome)
                codici[nome] = codici[gruppo[0]]
                trovato = True
                break
        if not trovato:
            codice = f"Persona_{len(gruppi)+1}"
            gruppi.append([nome])
            codici[nome] = codice
    return codici
 
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
 
        # Normalizza i nomi delle persone
        persone_uniche = list(set(entita['persone']))
        codici_persone = normalizza_nomi(persone_uniche, soglia=0.75)
        for persona in persone_uniche:
            codice = codici_persone[persona]
            # Il nodo è il codice, ma salvo anche il nome originale come attributo
            G.add_node(codice, tipo="Persona", nome_originale=persona)
            G.add_edge(doc_node, codice, relazione="contiene")

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
documento = "Doc_Fattura"  
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

# Connessione a Neo4j (modifica user, password e uri secondo la tua configurazione)
graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "Password"))

# Opzionale: cancella tutto prima di caricare
graph_db.delete_all()

# Carica nodi
for nodo, attr in G.nodes(data=True):
    label = attr.get('tipo', 'Entita')
    node = Node(label, name=str(nodo), **attr)
    graph_db.merge(node, label, "name")

# Carica archi
for sorgente, destinazione, attr in G.edges(data=True):
    sorgente_tipo = G.nodes[sorgente].get('tipo', 'Entita')
    destinazione_tipo = G.nodes[destinazione].get('tipo', 'Entita')
    node1 = graph_db.nodes.match(sorgente_tipo, name=str(sorgente)).first()
    node2 = graph_db.nodes.match(destinazione_tipo, name=str(destinazione)).first()
    if node1 and node2:
        relazione = attr.get('relazione', 'REL')
        rel = Relationship(node1, relazione, node2)
        for k, v in attr.items():
            rel[k] = v
        graph_db.merge(rel)