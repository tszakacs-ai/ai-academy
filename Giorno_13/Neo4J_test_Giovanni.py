from neo4j import GraphDatabase
from py2neo import Graph
 
uri ="bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j","Password123"))
 
with driver.session() as session:
    result = session.run("MATCH (n) RETURN n LIMIT 5")
    for record in result:
        print(record)
 
uri = "bolt://localhost:7687"
graph = Graph(uri, auth=("neo4j", "Password123"))
results = graph.run("MATCH (p:Persona) RETURN p.nome LIMIT 5")
 
for record in results:
    print(record)