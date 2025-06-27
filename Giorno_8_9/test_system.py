"""
Script di test e configurazione per il sistema di interrogazione PDF
Verifica connessioni, testa embeddings e mostra statistiche dell'indice
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, List

def test_azure_openai():
    """Test delle connessioni Azure OpenAI"""
    print("🔄 Test Azure OpenAI...")
    
    try:
        from openai import AzureOpenAI
        
        # Test Embedding client
        embedding_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
        )
        
        # Test embedding
        test_text = "Test di connessione per embedding"
        response = embedding_client.embeddings.create(
            input=test_text,
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        )
        
        embedding = response.data[0].embedding
        print(f"✅ Embedding client OK - Dimensioni: {len(embedding)}")
        
        # Test Chat client
        chat_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        )
        
        # Test chat completion
        chat_response = chat_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            messages=[{"role": "user", "content": "Rispondi semplicemente: OK"}],
            max_tokens=10
        )
        
        print(f"✅ Chat client OK - Risposta: {chat_response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Errore Azure OpenAI: {e}")
        return False

def test_pinecone_connection():
    """Test connessione Pinecone e analisi indice"""
    print("\n🔄 Test Pinecone...")
    
    try:
        from pinecone import Pinecone
        
        # Inizializza Pinecone con la nuova API
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Lista indici
        indexes = pc.list_indexes()
        available_indexes = [idx.name for idx in indexes]
        print(f"✅ Indici disponibili: {available_indexes}")
        
        # Verifica indice 'compdf'
        if 'compdf' in available_indexes:
            index = pc.Index('compdf')
            stats = index.describe_index_stats()
            
            print(f"✅ Indice 'compdf' trovato:")
            print(f"   - Dimensioni: {stats.dimension}")
            print(f"   - Vettori totali: {stats.total_vector_count}")
            print(f"   - Namespace: {list(stats.namespaces.keys()) if stats.namespaces else ['default']}")
            
            # Test query di esempio
            if stats.total_vector_count > 0:
                print("\n🔍 Test query di esempio...")
                test_vector = [0.1] * stats.dimension
                results = index.query(
                    vector=test_vector,
                    top_k=3,
                    include_metadata=True
                )
                
                print(f"✅ Query test completata: {len(results.matches)} risultati")
                for i, match in enumerate(results.matches[:2], 1):
                    metadata = match.metadata or {}
                    content_preview = str(metadata.get('text', metadata.get('content', 'N/A')))[:100]
                    print(f"   {i}. Score: {match.score:.4f} - Content: {content_preview}...")
            
            return True
        else:
            print(f"❌ Indice 'compdf' non trovato negli indici disponibili: {available_indexes}")
            return False
            
    except Exception as e:
        print(f"❌ Errore Pinecone: {e}")
        return False

def analyze_index_content():
    """Analizza il contenuto dell'indice per capire la struttura dei dati"""
    print("\n🔍 Analisi contenuto indice 'compdf'...")
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index('compdf')
        
        # Query per campioni di dati
        test_vector = [0.0] * 1536  # Assuming ADA-002 dimensions
        results = index.query(
            vector=test_vector,
            top_k=10,
            include_metadata=True
        )
        
        print(f"📊 Analisi di {len(results.matches)} documenti campione:")
        print("-" * 60)
        
        # Analizza metadati
        metadata_keys = set()
        sources = set()
        page_numbers = []
        
        for match in results.matches:
            if match.metadata:
                metadata_keys.update(match.metadata.keys())
                if 'source' in match.metadata:
                    sources.add(match.metadata['source'])
                if 'page_number' in match.metadata:
                    page_numbers.append(match.metadata['page_number'])
                elif 'page' in match.metadata:
                    page_numbers.append(match.metadata['page'])
        
        print(f"🔑 Chiavi metadata trovate: {sorted(list(metadata_keys))}")
        print(f"📄 Fonti documenti: {sorted(list(sources))}")
        if page_numbers:
            print(f"📖 Range pagine: {min(page_numbers)} - {max(page_numbers)}")
        
        # Mostra esempi di contenuto
        print(f"\n📝 Esempi di contenuto:")
        print("-" * 40)
        for i, match in enumerate(results.matches[:3], 1):
            metadata = match.metadata or {}
            content = metadata.get('text', metadata.get('content', 'Contenuto non disponibile'))
            source = metadata.get('source', 'Fonte sconosciuta')
            page = metadata.get('page_number', metadata.get('page', 'N/A'))
            
            print(f"{i}. Fonte: {source} (Pag. {page})")
            print(f"   Content: {str(content)[:150]}...")
            print(f"   Score: {match.score:.4f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Errore analisi indice: {e}")
        return False

def check_environment_variables():
    """Verifica che tutte le variabili d'ambiente necessarie siano configurate"""
    print("🔧 Verifica variabili d'ambiente...")
    
    required_vars = {
        'Azure Embedding': [
            'AZURE_EMBEDDING_ENDPOINT',
            'AZURE_EMBEDDING_API_KEY',
            'AZURE_EMBEDDING_DEPLOYMENT'
        ],
        'Azure Chat': [
            'AZURE_OPENAI_ENDPOINT', 
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_DEPLOYMENT'
        ],
        'Pinecone': [
            'PINECONE_API_KEY'
        ]
    }
    
    all_good = True
    
    for category, vars_list in required_vars.items():
        print(f"\n📋 {category}:")
        for var in vars_list:
            value = os.getenv(var)
            if value:
                # Mostra solo i primi e ultimi caratteri per sicurezza
                display_value = f"{value[:10]}...{value[-10:]}" if len(value) > 20 else value
                print(f"   ✅ {var}: {display_value}")
            else:
                print(f"   ❌ {var}: NON CONFIGURATA")
                all_good = False
    
    return all_good

def run_sample_queries():
    """Esegue alcune query di esempio per testare il sistema"""
    print("\n🚀 Test query di esempio...")
    
    try:
        # Import del sistema principale
        from pdf_query_system import PDFQuerySystem
        
        system = PDFQuerySystem()
        
        sample_queries = [
            "migrazione globale plastica",
            "regolamento 1935/2004",
            "decreto ministeriale 1973"
        ]
        
        for query in sample_queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 40)
            
            results = system.search_documents(query, top_k=3, score_threshold=0.5)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. Score: {result.score:.3f} - Fonte: {result.source}")
                    print(f"   Preview: {result.content[:100]}...")
            else:
                print("   Nessun risultato trovato")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test query: {e}")
        return False

def main():
    """Funzione principale per eseguire tutti i test"""
    print("🔬 SISTEMA DI TEST E CONFIGURAZIONE")
    print("=" * 60)
    
    # Carica variabili d'ambiente
    load_dotenv()
    
    # Test sequenza
    tests = [
        ("Variabili d'ambiente", check_environment_variables),
        ("Azure OpenAI", test_azure_openai),
        ("Pinecone", test_pinecone_connection),
        ("Analisi indice", analyze_index_content),
        ("Query di esempio", run_sample_queries)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Errore in {test_name}: {e}")
            results[test_name] = False
    
    # Riepilogo finale
    print(f"\n{'='*20} RIEPILOGO FINALE {'='*20}")
    all_passed = True
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 TUTTI I TEST SUPERATI!")
        print("Il sistema è pronto per l'uso. Puoi ora eseguire:")
        print("python pdf_query_system.py")
    else:
        print(f"\n⚠️  Alcuni test non sono riusciti. Verifica la configurazione.")

if __name__ == "__main__":
    main()