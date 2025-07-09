from neo4j import GraphDatabase
from py2neo import Graph

conn_url = "bolt://localhost:7687"
neo4j_driver = GraphDatabase.driver(conn_url, auth=("neo4j", "pass1"))

with neo4j_driver.session() as conn_session:
    query_result = conn_session.run("MATCH (n) RETURN n LIMIT 5")
    for nodo in query_result:
        print(nodo)

conn_graph = Graph(conn_url, auth=("neo4j", "pass1"))
persone = conn_graph.run("MATCH (p:Persona) RETURN p.nome LIMIT 5")

for persona in persone:
    print(persona)