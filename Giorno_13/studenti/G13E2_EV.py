# Analisi documenti e costruzione grafo entità-relazioni
import networkx as nx
import pandas as pd
import re
from transformers import pipeline
from pathlib import Path
import matplotlib.pyplot as plt


estrattore_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
grafo = nx.DiGraph()
cartella_documenti = pd.read_csv(r"C:\Users\MF579CW\OneDrive - EY\Desktop\EY_scripts\eai-academy\Giorno_5\studenti\docs")

for percorso_file in cartella_documenti.glob("*"):
    if percorso_file.is_file():
        with open(percorso_file, 'r', encoding='utf-8') as file_letto:
            testo_doc = file_letto.read()

        # diz per entità estratte
        diz_entita = {
            'nomi_persone': [],
            'societa': [],
            'date_rif': [],
            'valori_importo': [],
            'prodotti_servizi': [],
            'num_fatture': [],
            'localita': [],
            'iban_trovati': [],
            'cf_trovati': []
        }

        lista_entita = estrattore_ner(testo_doc)
        for elemento in lista_entita:
            tipo_ent = elemento['entity_group']
            testo_ent = elemento.get('word', '').strip()
            if not testo_ent:
                continue

            if tipo_ent == 'PER':
                diz_entita['nomi_persone'].append(testo_ent)
            elif tipo_ent == 'ORG':
                diz_entita['societa'].append(testo_ent)
            elif tipo_ent == 'DATE':
                diz_entita['date_rif'].append(testo_ent)
            elif tipo_ent == 'MISC' and re.search(r'\b\d{1,3}(?:\.\d{3})*,\d{2}\b|€', testo_ent):
                diz_entita['valori_importo'].append(testo_ent)
            elif tipo_ent in ['LOC', 'GPE']:
                diz_entita['localita'].append(testo_ent)


        regex_iban = r'[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}[A-Z0-9]{0,16}'
        regex_cf = r'[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]'
        regex_fattura = r'[nN]\.?\s?\d+'

        diz_entita['iban_trovati'].extend(re.findall(regex_iban, testo_doc))
        diz_entita['cf_trovati'].extend(re.findall(regex_cf, testo_doc))
        diz_entita['num_fatture'].extend(re.findall(regex_fattura, testo_doc))

        prodotti_estratti = re.findall(r"(?:per|riguardante|relativa a|oggetto:)\s+([\w\s\-]{3,40})(?=\.|,|\n|$)", testo_doc, re.IGNORECASE)
        diz_entita['prodotti_servizi'].extend(prodotti_estratti)

 
 
        nodo_doc = f"Doc_{percorso_file.stem}"
        grafo.add_node(nodo_doc, tipo="Documento")

   
        if "fattura" in testo_doc.lower():
            tipo_documento = "Fattura"
        elif "contratto" in testo_doc.lower():
            tipo_documento = "Contratto"
        elif "ordine" in testo_doc.lower():
            tipo_documento = "Ordine"
        elif "email" in testo_doc.lower() or "@gmail.com" in testo_doc:
            tipo_documento = "Email"
        else:
            tipo_documento = "Generico"
        grafo.nodes[nodo_doc]['tipo_documento'] = tipo_documento

   
        aziende_uniche = list(dict.fromkeys(diz_entita['societa']))
        if len(aziende_uniche) >= 2:
            mittente, ricevente = aziende_uniche[0], aziende_uniche[1]
            grafo.add_node(mittente, tipo="Azienda")
            grafo.add_edge(nodo_doc, mittente, relazione="emessa_da")
            grafo.add_node(ricevente, tipo="Azienda")
            grafo.add_edge(nodo_doc, ricevente, relazione="inviata_a")
        else:
            for azienda in aziende_uniche:
                grafo.add_node(azienda, tipo="Azienda")
                grafo.add_edge(nodo_doc, azienda, relazione="coinvolge")

        for persona in set(diz_entita['nomi_persone']):
            grafo.add_node(persona, tipo="Persona")
            grafo.add_edge(nodo_doc, persona, relazione="contiene")

        for data in set(diz_entita['date_rif']):
            grafo.add_node(data, tipo="Data")
            grafo.add_edge(nodo_doc, data, relazione="data_documento")

        for importo in set(diz_entita['valori_importo']):
            grafo.add_node(importo, tipo="Importo")
            grafo.add_edge(nodo_doc, importo, relazione="importo_totale")

        for num_fatt in set(diz_entita['num_fatture']):
            grafo.add_node(num_fatt, tipo="NumeroFattura")
            grafo.add_edge(nodo_doc, num_fatt, relazione="identificata_da")

        for iban in set(diz_entita['iban_trovati']):
            grafo.add_node(iban, tipo="IBAN")
            grafo.add_edge(nodo_doc, iban, relazione="pagabile_su")

        for cf in set(diz_entita['cf_trovati']):
            grafo.add_node(cf, tipo="CodFiscale")
            grafo.add_edge(nodo_doc, cf, relazione="cf_riferito")

        for prodotto in set(diz_entita['prodotti_servizi']):
            grafo.add_node(prodotto.strip(), tipo="Prodotto")
            grafo.add_edge(nodo_doc, prodotto.strip(), relazione="oggetto")

        for luogo in set(diz_entita['localita']):
            grafo.add_node(luogo, tipo="Luogo")
            grafo.add_edge(nodo_doc, luogo, relazione="localizzato_in")

parole_stop = {'di', 'in', 'per', 'a', 'il', 'la', 'e'}
nodi_da_eliminare = [n for n in grafo.nodes() if isinstance(n, str) and (len(n.strip()) < 2 or n.lower() in parole_stop)]
for nodo in nodi_da_eliminare:
    grafo.remove_node(nodo)

# stats grafo
print(f"\nGrafo creato con {grafo.number_of_nodes()} nodi e {grafo.number_of_edges()} archi")

print("\nConteggio tipi di nodi:")
conteggio_tipologie = {}
for nodo, attributi in grafo.nodes(data=True):
    tipo_nodo = attributi.get('tipo', 'Sconosciuto')
    conteggio_tipologie[tipo_nodo] = conteggio_tipologie.get(tipo_nodo, 0) + 1
for tipo, quanti in conteggio_tipologie.items():
    print(f"- {tipo}: {quanti}")

doc_esempio = "Doc_fattura1"
if grafo.has_node(doc_esempio):
    print(f"\nRelazioni da: {doc_esempio}")
    tipo_doc = grafo.nodes[doc_esempio].get("tipo_documento", "N/A")
    print(f"Tipo documento: {tipo_doc}")
    for successivo in grafo.successors(doc_esempio):
        relazione = grafo[doc_esempio][successivo].get('relazione', 'relazione_sconosciuta')
        tipo_succ = grafo.nodes[successivo].get('tipo', 'TipoSconosciuto')
        print(f"- {relazione}: {successivo} ({tipo_succ})")
else:
    print(f"Documento '{doc_esempio}' non trovato.")






#grafo
plt.figure(figsize=(14, 12))
posizioni = nx.spring_layout(grafo, k=1.5, iterations=100)
etichette = {n: (n[:12] + "...") if len(n) > 15 else n for n in grafo.nodes()}

nx.draw_networkx_nodes(grafo, posizioni, node_color='skyblue', node_size=800)
nx.draw_networkx_edges(grafo, posizioni, arrowstyle='->', arrowsize=15, edge_color='gray')
nx.draw_networkx_labels(grafo, posizioni, labels=etichette, font_size=8)

etichette_archi = nx.get_edge_attributes(grafo, 'relazione')
nx.draw_networkx_edge_labels(grafo, posizioni, edge_labels=etichette_archi, font_color='red', font_size=7)

plt.title("Grafo Entità-Relazioni", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()