# main.py - Knowledge Graph Aziendale
# UNICO FILE NECESSARIO

from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, password):
        # Connessione standard a Neo4j Desktop
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))
        
    def close(self):
        self.driver.close()
    
    def inizializza_tutto(self):
        """Fa tutto: pulisce, crea vincoli, carica dati e relazioni"""
        with self.driver.session() as session:
            
            # 1. Pulisce tutto
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database pulito")
            
            # 2. Crea vincoli
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Persona) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Azienda) REQUIRE a.partita_iva IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Documento) REQUIRE d.id IS UNIQUE")
            print("✅ Vincoli creati")
            
            # 3. Carica tutto insieme (nodi + relazioni)
            session.run("""
                // Crea persone
                CREATE (mario:Persona {id: 'P001', nome: 'Mario', cognome: 'Rossi', ruolo: 'Manager'})
                CREATE (anna:Persona {id: 'P002', nome: 'Anna', cognome: 'Verdi', ruolo: 'Developer'})
                CREATE (luigi:Persona {id: 'P003', nome: 'Luigi', cognome: 'Bianchi', ruolo: 'Consulente'})
                
                // Crea aziende
                CREATE (tech:Azienda {partita_iva: '12345001', nome: 'Tech Solutions SRL', settore: 'IT'})
                CREATE (innova:Azienda {partita_iva: '12345002', nome: 'Innovate Corp', settore: 'Consulting'})
                
                // Crea progetti
                CREATE (prog1:Progetto {id: 'PR001', nome: 'Digitalizzazione', budget: 50000})
                CREATE (prog2:Progetto {id: 'PR002', nome: 'AI Project', budget: 75000})
                
                // Crea documenti
                CREATE (doc1:Documento {id: 'DOC001', tipo: 'Contratto', importo: 25000, stato: 'Firmato'})
                CREATE (doc2:Documento {id: 'DOC002', tipo: 'Proposta', importo: 15000, stato: 'In attesa'})
                
                // Crea fatture
                CREATE (fat1:Fattura {numero: 'F001', importo: 12000, stato: 'Pagata'})
                CREATE (fat2:Fattura {numero: 'F002', importo: 8000, stato: 'In attesa'})
                CREATE (fat3:Fattura {numero: 'F003', importo: 15000, stato: 'Pagata'})
                
                // Crea relazioni
                CREATE (mario)-[:HA_FIRMATO {data: '2024-01-15'}]->(doc1)
                CREATE (luigi)-[:HA_FIRMATO {data: '2024-02-01'}]->(doc2)
                CREATE (mario)-[:PARTECIPA_A {ruolo: 'Project Manager'}]->(prog1)
                CREATE (anna)-[:PARTECIPA_A {ruolo: 'Developer'}]->(prog1)
                CREATE (anna)-[:PARTECIPA_A {ruolo: 'Lead Developer'}]->(prog2)
                CREATE (fat1)-[:EMESSA_DA]->(tech)
                CREATE (fat2)-[:EMESSA_DA]->(innova)
                CREATE (fat3)-[:EMESSA_DA]->(tech)
                CREATE (doc1)-[:RIFERITO_A]->(tech)
            """)
            print("✅ Dati e relazioni caricati")
    
    def query_contratti_firmati(self):
        """Chi ha firmato contratti?"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Persona)-[r:HA_FIRMATO]->(d:Documento)
                WHERE d.tipo = 'Contratto'
                RETURN p.nome + ' ' + p.cognome AS persona, d.id AS documento, d.importo AS importo
            """)
            return list(result)
    
    def query_fatture_alte(self, soglia=10000):
        """Fatture sopra una soglia"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:Fattura)
                WHERE f.importo > $soglia
                RETURN f.numero AS fattura, f.importo AS importo, f.stato AS stato
                ORDER BY f.importo DESC
            """, soglia=soglia)
            return list(result)
    
    def query_progetti_persona(self, nome_persona):
        """Progetti di una persona"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Persona)-[r:PARTECIPA_A]->(pr:Progetto)
                WHERE p.nome = $nome
                RETURN pr.nome AS progetto, r.ruolo AS ruolo, pr.budget AS budget
            """, nome=nome_persona)
            return list(result)
    
    def statistiche(self):
        """Conta tutti i nodi"""
        statistiche = {}
        with self.driver.session() as session:
            # Conta ogni tipo separatamente
            queries = {
                'Persone': "MATCH (p:Persona) RETURN count(p) as count",
                'Aziende': "MATCH (a:Azienda) RETURN count(a) as count",
                'Documenti': "MATCH (d:Documento) RETURN count(d) as count",
                'Progetti': "MATCH (pr:Progetto) RETURN count(pr) as count",
                'Fatture': "MATCH (f:Fattura) RETURN count(f) as count",
                'Relazioni': "MATCH ()-[r]->() RETURN count(r) as count"
            }
            
            for nome, query in queries.items():
                result = session.run(query)
                statistiche[nome] = result.single()['count']
            
            return statistiche

# ESECUZIONE PRINCIPALE
if __name__ == "__main__":
    
    # STEP 1: INSERISCI LA TUA PASSWORD QUI
    PASSWORD_NEO4J = "123" 
    
    print("🚀 Avvio Knowledge Graph...")
    
    try:
        # STEP 2: Connetti e inizializza
        kg = KnowledgeGraph(PASSWORD_NEO4J)
        kg.inizializza_tutto()
        
        # STEP 3: Mostra statistiche
        print("\n📊 STATISTICHE:")
        stats = kg.statistiche()
        for tipo, numero in stats.items():
            print(f"  {tipo}: {numero}")
        
        # STEP 4: Esegui query di esempio
        print("\n🔍 CHI HA FIRMATO CONTRATTI:")
        contratti = kg.query_contratti_firmati()
        for record in contratti:
            print(f"  {record['persona']} → {record['documento']} (€{record['importo']:,})")
        
        print("\n💰 FATTURE SOPRA €10.000:")
        fatture = kg.query_fatture_alte(10000)
        for record in fatture:
            print(f"  {record['fattura']}: €{record['importo']:,} - {record['stato']}")
        
        print("\n🎯 PROGETTI DI MARIO:")
        progetti = kg.query_progetti_persona("Mario")
        for record in progetti:
            print(f"  {record['progetto']}: {record['ruolo']} (budget €{record['budget']:,})")
        
        print("\n✅ COMPLETATO!")
        print("🌐 Apri http://localhost:7474 per vedere il grafo visualmente")
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        if "authentication" in str(e).lower():
            print("💡 Controlla la password Neo4j nella riga 'PASSWORD_NEO4J'")
        elif "connection" in str(e).lower():
            print("💡 Assicurati che Neo4j sia avviato nel Desktop")
        else:
            print("💡 Verifica che Neo4j Desktop sia installato e running")
    
    finally:
        kg.close()