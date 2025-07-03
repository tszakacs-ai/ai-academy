query="""
CREATE (c:Cliente {nome : $nome})
CREATE (co: Contratto {anno_scadenza : $anno:scadenza})
CREATE (c)-[:HA IL CONTRATTO]->(co)
"""

query="""
MATCH (c:Cliente) -[: HA IL CONTRATTO]-> (co: contratto)
WHERE co.anno_scadenza>2024
RETURN c.nome, c.cognome
"""

prompt="""
Ti invierò la risposta di una query Cypher. Traduci la risposta in un testo in italiano, 
come se stessi parlando con un cliente e scrivila sotto formato di email o messaggio per potrela inviare.
Risposta :{risposta}
"""
