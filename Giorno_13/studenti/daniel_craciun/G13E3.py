from neo4j import GraphDatabase
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))
with driver.session() as session:
    result = session.run("MATCH (n) RETURN n LIMIT 5")
for record in result:
    print(record)