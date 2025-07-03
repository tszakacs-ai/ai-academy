query = """
CREATE (c:Cliente {nome: $nome})
CREATE (f: Fattura {numero: $num_fattura})
CREATE (c)-[HA_RICEVUTO] -> (f)
"""


query = """ 
MATCH (f: Fattura)
WHERE f.importo > 1000
RETURN f.numero, f.importo
"""

prompt = """
    Domanda business: Quali fatture hanno un importo superiore a 1000 euro?

    Risultato della query:
    - Fattura INV2025: 1500 euro
    - Fattura INV2026: 2500 euro

    Riformula la risposta in linguaggio naturale, chiaro e sintetico per un manager. """
