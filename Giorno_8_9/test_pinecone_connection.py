"""
Test di connessione a Pinecone
"""
import os
from dotenv import load_dotenv

def test_pinecone_connection():
    """Test della connessione a Pinecone"""
    load_dotenv()
    
    # Importa Pinecone
    try:
        from pdf_query_system import Pinecone
        print("✅ Pinecone library importata")
    except ImportError:
        print("❌ Pinecone non installato. Esegui: pip install pinecone")
        return False
    
    # Leggi credenziali
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "quickstart")
    
    if not api_key:
        print("❌ PINECONE_API_KEY non trovata nel file .env")
        return False
    
    print("=== TEST CONNESSIONE PINECONE ===")
    print(f"API Key: {api_key[:20]}...")
    print(f"Index Name: {index_name}")
    
    try:
        # Inizializza Pinecone
        print("🔄 Inizializzazione Pinecone...")
        pc = Pinecone(api_key=api_key)
        
        # Lista degli indici
        print("🔄 Lista indici...")
        indexes = pc.list_indexes()
        print(f"✅ Indici disponibili: {[idx.name for idx in indexes]}")
        
        # Connetti all'indice
        if index_name in [idx.name for idx in indexes]:
            print(f"🔄 Connessione all'indice '{index_name}'...")
            index = pc.Index(index_name)
            
            # Ottieni statistiche dell'indice
            print("🔄 Statistiche indice...")
            stats = index.describe_index_stats()
            print(f"✅ Statistiche indice:")
            print(f"   - Dimensioni: {stats.dimension}")
            print(f"   - Vettori totali: {stats.total_vector_count}")
            print(f"   - Namespace: {stats.namespaces}")
            
            # Test semplice di query
            if stats.total_vector_count > 0:
                print("🔍 Test query di esempio...")
                # Query con un vettore di test
                test_vector = [0.1] * stats.dimension
                results = index.query(
                    vector=test_vector,
                    top_k=3,
                    include_metadata=True
                )
                
                print(f"✅ Query completata: {len(results.matches)} risultati")
                for i, match in enumerate(results.matches[:2]):
                    print(f"   {i+1}. ID: {match.id}, Score: {match.score:.4f}")
                    if match.metadata:
                        content = match.metadata.get('content', 'N/A')
                        print(f"      Content: {content[:100]}...")
            else:
                print("⚠️  Indice vuoto - nessun vettore presente")
                
                # Test di upsert
                print("📝 Test inserimento documento...")
                test_doc = {
                    "id": "test_001",
                    "values": [0.1] * stats.dimension,
                    "metadata": {
                        "content": "Test document for connection",
                        "type": "test"
                    }
                }
                
                upsert_result = index.upsert(vectors=[test_doc])
                print(f"✅ Upsert completato: {upsert_result}")
                
                # Verifica inserimento
                fetch_result = index.fetch(ids=["test_001"])
                if "test_001" in fetch_result.vectors:
                    print("✅ Documento di test trovato")
                    
                    # Pulizia
                    index.delete(ids=["test_001"])
                    print("🧹 Documento di test eliminato")
            
            return True
            
        else:
            print(f"❌ Indice '{index_name}' non trovato")
            print("Indici disponibili:", [idx.name for idx in indexes])
            return False
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_compliance_index():
    """Crea un indice per il progetto compliance se non esiste"""
    load_dotenv()
    
    from pdf_query_system import Pinecone, ServerlessSpec
    
    api_key = os.getenv("PINECONE_API_KEY")
    compliance_index = "compliance-chunks"
    
    try:
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        existing_names = [idx.name for idx in indexes]
        
        if compliance_index not in existing_names:
            print(f"📝 Creazione nuovo indice '{compliance_index}'...")
            
            pc.create_index(
                name=compliance_index,
                dimension=1536,  # Per embeddings ada-002
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            print(f"✅ Indice '{compliance_index}' creato con successo!")
            
            # Attendi che l'indice sia pronto
            import time
            print("⏳ Attendo che l'indice sia pronto...")
            time.sleep(10)
            
            return compliance_index
        else:
            print(f"✅ Indice '{compliance_index}' già esistente")
            return compliance_index
            
    except Exception as e:
        print(f"❌ Errore nella creazione dell'indice: {e}")
        return None

def test_embeddings_compatibility():
    """Test compatibilità con embeddings Azure OpenAI"""
    load_dotenv()
    
    from pdf_query_system import Pinecone
    from openai import AzureOpenAI
    
    # Test Azure OpenAI embeddings
    print("\n=== TEST COMPATIBILITÀ EMBEDDINGS ===")
    
    try:
        # Setup Azure OpenAI
        azure_client = AzureOpenAI(
            azure_endpoint=os.getenv("ADA_ENDPOINT"),
            api_key=os.getenv("ADA_API_KEY"),
            api_version=os.getenv("ADA_API_VERSION", "2024-02-01")
        )
        
        # Test embedding
        test_text = "Questo è un test per verificare la compatibilità degli embeddings"
        
        print("🔄 Generazione embedding di test...")
        response = azure_client.embeddings.create(
            input=test_text,
            model=os.getenv("ADA_DEPLOYMENT_NAME", "text-embedding-ada-002")
        )
        
        embedding = response.data[0].embedding
        print(f"✅ Embedding generato: dimensioni {len(embedding)}")
        
        # Test con Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "quickstart")
        index = pc.Index(index_name)
        
        # Test upsert con embedding reale
        print("📝 Test upsert con embedding Azure...")
        test_doc = {
            "id": "azure_embedding_test",
            "values": embedding,
            "metadata": {
                "content": test_text,
                "source": "azure_openai_test",
                "type": "compatibility_test"
            }
        }
        
        upsert_result = index.upsert(vectors=[test_doc])
        print(f"✅ Upsert con embedding Azure completato")
        
        # Test query
        print("🔍 Test query con stesso embedding...")
        query_results = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            filter={"type": "compatibility_test"}
        )
        
        if query_results.matches:
            match = query_results.matches[0]
            print(f"✅ Query riuscita: ID {match.id}, Score: {match.score:.4f}")
        
        # Pulizia
        index.delete(ids=["azure_embedding_test"])
        print("🧹 Test document eliminato")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test compatibilità: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Test completo Pinecone per progetto Compliance")
    
    # Test 1: Connessione base
    connection_ok = test_pinecone_connection()
    
    if connection_ok:
        print("\n" + "="*50)
        
        # Test 2: Crea indice per compliance se necessario
        compliance_index = create_compliance_index()
        
        if compliance_index:
            print("\n" + "="*50)
            
            # Test 3: Compatibilità embeddings
            embedding_ok = test_embeddings_compatibility()
            
            if embedding_ok:
                print("\n🎉🎉🎉 PERFETTO! 🎉🎉🎉")
                print("✅ Pinecone connesso")
                print("✅ Indice compliance pronto")
                print("✅ Embeddings Azure compatibili")
                print(f"✅ Indice da usare: {compliance_index}")
                print("\n🚀 Tutto pronto per il caricamento dei dati!")
            else:
                print("\n⚠️  Connessione OK ma problemi con embeddings")
        else:
            print("\n⚠️  Connessione OK ma problemi con creazione indice")
    else:
        print("\n❌ Problemi di connessione con Pinecone")
        print("Verifica:")
        print("1. API key corretta nel .env")
        print("2. Connessione internet")
        print("3. Pinecone installato: pip install pinecone")