# main_final.py - Knowledge Graph dai tuoi documenti (VERSIONE SEMPLIFICATA)

from neo4j import GraphDatabase
import re
import os
from pathlib import Path
from transformers import pipeline

# REGEX PATTERNS PER ESTRAZIONE
PATTERNS = {
    'PARTITA_IVA': r'\b\d{11}\b',
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'IMPORTO': r'€\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*€',
    'TELEFONO': r'\b(?:\+39\s?)?(?:0\d{1,4}\s?)?\d{6,10}\b'
}

class KnowledgeGraphSemplice:
    def __init__(self, password):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))
        print("📡 Caricamento modello NER...")
        self.ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        print("✅ Modello caricato")
        
    def close(self):
        self.driver.close()
    
    def leggi_documenti(self):
        """Legge tutti i file .txt dalla cartella sample_documents"""
        documenti = []
        cartella = Path("sample_documents")
        
        for file_txt in cartella.glob("*.txt"):
            with open(file_txt, 'r', encoding='utf-8') as f:
                contenuto = f.read()
                documenti.append(contenuto)
                print(f"📄 Letto: {file_txt.name}")
        
        return documenti
    
    def estrai_entita(self, testo):
        """Estrae entità con regex + NER"""
        entita = {}
        
        # Regex
        for label, pattern in PATTERNS.items():
            matches = re.findall(pattern, testo, re.IGNORECASE)
            if matches:
                entita[label] = list(set(matches))
        
        # NER per persone e organizzazioni
        try:
            ner_results = self.ner(testo)
            for ent in ner_results:
                if ent['score'] > 0.7:
                    label = ent['entity_group']
                    if label in ['PER', 'PERSON', 'ORG']:
                        if label not in entita:
                            entita[label] = []
                        word = ent['word'].replace('##', '')
                        if word not in entita[label]:
                            entita[label].append(word)
        except:
            pass
        
        return entita
    
    def crea_knowledge_graph(self):
        """Processo completo: leggi documenti → estrai entità → crea grafo"""
        
        # 1. Pulisci e prepara database
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Persona) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Azienda) REQUIRE a.partita_iva IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Documento) REQUIRE d.id IS UNIQUE")
        print("✅ Database preparato")
        
        # 2. Leggi documenti
        documenti = self.leggi_documenti()
        print(f"✅ {len(documenti)} documenti caricati")
        
        # 3. Elabora ogni documento
        persona_id = 1
        azienda_id = 1
        doc_id = 1
        
        for testo in documenti:
            print(f"🔍 Elaboro documento {doc_id}...")
            entita = self.estrai_entita(testo)
            
            with self.driver.session() as session:
                
                # Crea documento
                session.run("""
                    MERGE (d:Documento {id: $doc_id})
                    SET d.tipo = $tipo, d.testo = $testo, d.importo = $importo
                """, 
                doc_id=f"DOC{doc_id:03d}",
                tipo=self.inferisci_tipo(testo),
                testo=testo[:200] + "..." if len(testo) > 200 else testo,
                importo=self.estrai_importo(entita))
                
                # Crea persone
                if 'PER' in entita or 'PERSON' in entita:
                    persone = entita.get('PER', []) + entita.get('PERSON', [])
                    for nome in persone:
                        if len(nome) > 2:
                            session.run("""
                                MERGE (p:Persona {id: $persona_id})
                                SET p.nome = $nome, p.email = $email
                            """,
                            persona_id=f"P{persona_id:03d}",
                            nome=nome,
                            email=entita.get('EMAIL', [None])[0])
                            
                            # Relazione HA_FIRMATO se è un contratto
                            if 'contratto' in testo.lower():
                                session.run("""
                                    MATCH (p:Persona {id: $persona_id}), (d:Documento {id: $doc_id})
                                    MERGE (p)-[:HA_FIRMATO]->(d)
                                """, persona_id=f"P{persona_id:03d}", doc_id=f"DOC{doc_id:03d}")
                            
                            persona_id += 1
                
                # Crea aziende
                if 'ORG' in entita:
                    for org in entita['ORG']:
                        if len(org) > 2:
                            piva = entita.get('PARTITA_IVA', [f"87654321{azienda_id:03d}"])[0]
                            session.run("""
                                MERGE (a:Azienda {partita_iva: $piva})
                                SET a.nome = $nome, a.settore = $settore
                            """,
                            piva=piva,
                            nome=org,
                            settore=self.inferisci_settore(testo))
                            
                            # Relazione RIFERITO_A
                            session.run("""
                                MATCH (d:Documento {id: $doc_id}), (a:Azienda {partita_iva: $piva})
                                MERGE (d)-[:RIFERITO_A]->(a)
                            """, doc_id=f"DOC{doc_id:03d}", piva=piva)
                            
                            azienda_id += 1
            
            doc_id += 1
        
        print("✅ Knowledge Graph creato!")
    
    def mostra_risultati(self):
        """Mostra statistiche e query principali"""
        with self.driver.session() as session:
            
            # Statistiche
            print("\n📊 STATISTICHE:")
            stats_queries = {
                'Persone': "MATCH (p:Persona) RETURN count(p) as count",
                'Aziende': "MATCH (a:Azienda) RETURN count(a) as count", 
                'Documenti': "MATCH (d:Documento) RETURN count(d) as count",
                'Relazioni': "MATCH ()-[r]->() RETURN count(r) as count"
            }
            
            for nome, query in stats_queries.items():
                result = session.run(query).single()
                print(f"  {nome}: {result['count']}")
            
            # Query principali
            print("\n🔍 PERSONE NEI DOCUMENTI:")
            result = session.run("MATCH (p:Persona) RETURN p.nome as nome, p.email as email")
            for record in result:
                email = f" ({record['email']})" if record['email'] else ""
                print(f"  👤 {record['nome']}{email}")
            
            print("\n🏢 AZIENDE NEI DOCUMENTI:")
            result = session.run("MATCH (a:Azienda) RETURN a.nome as nome, a.settore as settore")
            for record in result:
                print(f"  🏢 {record['nome']} - {record['settore']}")
            
            print("\n📋 CONTRATTI FIRMATI:")
            result = session.run("""
                MATCH (p:Persona)-[:HA_FIRMATO]->(d:Documento)
                RETURN p.nome as persona, d.tipo as tipo, d.importo as importo
            """)
            for record in result:
                importo = f"€{record['importo']:,}" if record['importo'] else "N/A"
                print(f"  ✍️ {record['persona']} → {record['tipo']} ({importo})")
    
    # METODI DI SUPPORTO
    def inferisci_tipo(self, testo):
        testo_lower = testo.lower()
        if 'contratto' in testo_lower: return 'Contratto'
        elif 'fattura' in testo_lower: return 'Fattura'
        elif 'email' in testo_lower or 'comunicazione' in testo_lower: return 'Email'
        else: return 'Documento'
    
    def inferisci_settore(self, testo):
        testo_lower = testo.lower()
        if any(word in testo_lower for word in ['tech', 'software', 'it', 'digitale']): return 'IT'
        elif any(word in testo_lower for word in ['consulenza', 'servizi']): return 'Consulting'
        else: return 'Generale'
    
    def estrai_importo(self, entita):
        if 'IMPORTO' in entita:
            importo_str = entita['IMPORTO'][0]
            numeri = re.findall(r'\d+', importo_str.replace('.', '').replace(',', ''))
            if numeri:
                return int(''.join(numeri[:2]))
        return None

# ESECUZIONE PRINCIPALE
if __name__ == "__main__":
    
    # INSERISCI LA TUA PASSWORD NEO4J QUI
    PASSWORD_NEO4J = "123"  
    
    print("🚀 Knowledge Graph dai tuoi documenti")
    print("="*50)
    
    try:
        kg = KnowledgeGraphSemplice(PASSWORD_NEO4J)
        kg.crea_knowledge_graph()
        kg.mostra_risultati()
        
        print(f"\n✅ COMPLETATO!")
        print("🌐 Visualizza su: http://localhost:7474")
        print("🔍 Query: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50")
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        if "password" in str(e).lower():
            print("💡 Controlla la password Neo4j")
        elif "connection" in str(e).lower():
            print("💡 Avvia Neo4j Desktop")
        
    finally:
        kg.close()