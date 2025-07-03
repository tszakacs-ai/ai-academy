import re
from pathlib import Path
from transformers import pipeline
import networkx as nx
from py2neo import Graph, Node, Relationship

# 1. Definizione struttura grafo
ENTITA = {
    'PER': 'Persona',
    'ORG': 'Azienda',
    'LOC': 'Luogo',
    'GPE': 'Luogo',
    'MISC': 'Misc'
}

def normalizza_nomi(nomi, soglia=0.8):
    import difflib
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
            codice = f"ID_PERSONA_{len(gruppi)+1:03d}"
            gruppi.append([nome])
            codici[nome] = codice
    return codici

# 2. Pipeline di anonimizzazione
def estrai_entita(testo):
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    entities = ner(testo)
    entita = {'persone': [], 'aziende': [], 'luoghi': [], 'iban': []}
    for ent in entities:
        label = ent['entity_group']
        word = ent.get('word', '').strip()
        if label == 'PER':
            entita['persone'].append(word)
        elif label == 'ORG':
            entita['aziende'].append(word)
        elif label in ['LOC', 'GPE']:
            entita['luoghi'].append(word)
    # IBAN via regex
    iban_pattern = r'[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}[A-Z0-9]{0,16}'
    entita['iban'].extend(re.findall(iban_pattern, testo))
    return entita

def anonimizza_testo(testo, mapping):
    for originale, anonimo in mapping.items():
        testo = re.sub(re.escape(originale), anonimo, testo)
    return testo

# 3. Ripristino dei dati
def ripristina_testo(testo_anon, reverse_mapping):
    for anonimo, originale in reverse_mapping.items():
        testo_anon = re.sub(re.escape(anonimo), originale, testo_anon)
    return testo_anon

# 4. Esempio di pipeline
def main():
    # Simulazione: carica un testo aziendale
    testo = """
    Mario Rossi ha inviato un pagamento di 1000€ a IT60X0542811101000000123456 per conto di ACME S.p.A. L'ufficio è a Milano.
    """
    entita = estrai_entita(testo)
    persone = list(set(entita['persone']))
    aziende = list(set(entita['aziende']))
    luoghi = list(set(entita['luoghi']))
    ibans = list(set(entita['iban']))

    # Normalizza persone
    codici_persone = normalizza_nomi(persone)
    mapping = {**codici_persone}
    mapping.update({iban: f"ID_IBAN_{i+1:03d}" for i, iban in enumerate(ibans)})
    mapping.update({az: f"ID_AZIENDA_{i+1:03d}" for i, az in enumerate(aziende)})
    mapping.update({luogo: f"ID_LUOGO_{i+1:03d}" for i, luogo in enumerate(luoghi)})

    testo_anon = anonimizza_testo(testo, mapping)
    print("Testo anonimizzato:\n", testo_anon)

    # Crea grafo NetworkX
    G = nx.DiGraph()
    for persona, codice in codici_persone.items():
        G.add_node(codice, tipo="Persona", nome=persona)
    for az, codice in zip(aziende, [mapping[a] for a in aziende]):
        G.add_node(codice, tipo="Azienda", nome=az)
    for luogo, codice in zip(luoghi, [mapping[l] for l in luoghi]):
        G.add_node(codice, tipo="Luogo", nome=luogo)
    for iban, codice in zip(ibans, [mapping[i] for i in ibans]):
        G.add_node(codice, tipo="IBAN", iban=iban)
    # Relazioni esempio
    if persone and aziende:
        G.add_edge(codici_persone[persone[0]], mapping[aziende[0]], relazione="lavora_per")
    if persone and ibans:
        G.add_edge(codici_persone[persone[0]], mapping[ibans[0]], relazione="possiede_iban")
    if aziende and luoghi:
        G.add_edge(mapping[aziende[0]], mapping[luoghi[0]], relazione="localizzata_in")

    # Carica su Neo4j
    graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "Password2"))  # Cambia password!
    graph_db.delete_all()
    for nodo, attr in G.nodes(data=True):
        label = attr.get('tipo', 'Entita')
        node = Node(label, name=str(nodo), **attr)
        graph_db.merge(node, label, "name")
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

    # Ripristino (lookup)
    reverse_mapping = {v: k for k, v in mapping.items()}
    testo_ripristinato = ripristina_testo(testo_anon, reverse_mapping)
    print("\nTesto ripristinato:\n", testo_ripristinato)

if __name__ == "__main__":
    main()