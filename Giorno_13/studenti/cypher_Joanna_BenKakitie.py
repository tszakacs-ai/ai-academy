query = """
CREATE (c:Client {nome: $nome})
CREATE (f:Fattura {nome: $fattura})
CREATE (c)-[:HA_RICEVUTO]->(f)
"""

query = """
MATCH (f:Fattura)
WHERE f.importo > 1000
RETURN f.numero, f.importo
"""

prompt = """

"""