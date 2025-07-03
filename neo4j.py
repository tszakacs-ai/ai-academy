from py2neo import Graph, Node, Relationship

graph= Graph("neo4j://127.0.0.1:7687", auth=("neo4j", "12341234"))

persona = Node("Persona", nome="Mario", cognome="Rossi")
documento = Node("Documento", tipo="Fattura", anno=2024)
azienda = Node("Azienda", nome="Tech Solutions")

graph.create(persona)
graph.create(documento)
graph.create(azienda)

rel1 = Relationship(persona, "HA_FIRMATO", documento)
rel2= Relationship(documento, "RAPPRESENTA", azienda)
graph.create(rel1)

query="""
MATCH (p:Persona)-[:HA_FIRMATO]->(d:Documento)
WHERE d.anno = 2024
RETURN p.nome, p.cognome, d.tipo    
"""
result = graph.run(query)

for record in result:
    print(f"Nome: {record['p.nome']}, Cognome: {record['p.cognome']}, Tipo Documento: {record['d.tipo']}")