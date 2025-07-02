"""
Script di test per verificare le connessioni e le funzionalità del sistema RAG
"""

import os
from dotenv import load_dotenv
import unittest
from typing import Dict, Any, List

# Carica variabili d'ambiente
load_dotenv()

class TestAzureConnections(unittest.TestCase):
    """Test per le connessioni ai servizi Azure"""
    
    def test_azure_openai_chat(self):
        """Verifica la connessione ad Azure OpenAI per Chat"""
        from openai import AzureOpenAI
        
        try:
            # Crea client
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            # Test semplice
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Say 'Hello, test passed!'"}],
                max_tokens=10
            )
            
            # Verifica risposta
            self.assertIsNotNone(response)
            self.assertTrue(len(response.choices) > 0)
            print("✅ Azure OpenAI Chat: connessione OK")
            
        except Exception as e:
            self.fail(f"❌ Azure OpenAI Chat: errore - {e}")
    
    def test_azure_openai_embedding(self):
        """Verifica la connessione ad Azure OpenAI per Embeddings"""
        from openai import AzureOpenAI
        
        try:
            # Crea client
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            # Test semplice
            deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            response = client.embeddings.create(
                input="Test text for embedding",
                model=deployment
            )
            
            # Verifica risposta
            self.assertIsNotNone(response)
            self.assertTrue(len(response.data) > 0)
            self.assertTrue(len(response.data[0].embedding) > 0)
            print("✅ Azure OpenAI Embedding: connessione OK")
            
        except Exception as e:
            self.fail(f"❌ Azure OpenAI Embedding: errore - {e}")

class TestPineconeConnection(unittest.TestCase):
    """Test per la connessione a Pinecone"""
    
    def test_pinecone_index(self):
        """Verifica la connessione a Pinecone e accesso all'indice"""
        from pinecone import Pinecone
        
        try:
            # Inizializza Pinecone
            api_key = os.getenv("PINECONE_API_KEY")
            pc = Pinecone(api_key=api_key)
            
            # Verifica lista indici
            indexes = pc.list_indexes()
            self.assertTrue(isinstance(indexes, list))
            
            # Prova ad accedere all'indice
            index_name = os.getenv("PINECONE_INDEX_NAME", "compliance50")
            index = pc.Index(index_name)
            
            # Verifica dimensioni indice
            stats = index.describe_index_stats()
            self.assertIsNotNone(stats)
            self.assertIn('dimension', stats)
            self.assertIn('namespaces', stats)
            self.assertIn('total_vector_count', stats)
            
            print(f"✅ Pinecone: connessione OK - Indice '{index_name}' trovato")
            print(f"   Dimensione vettori: {stats['dimension']}")
            print(f"   Totale vettori: {stats['total_vector_count']}")
            
        except Exception as e:
            self.fail(f"❌ Pinecone: errore - {e}")

class TestRAGSystem(unittest.TestCase):
    """Test per il sistema RAG"""
    
    def setUp(self):
        """Setup per i test del sistema RAG"""
        from src.config import setup_clients, AppConfig
        
        # Inizializza i client e la configurazione
        self.clients = setup_clients()
        self.config = AppConfig()
    
    def test_search_functionality(self):
        """Verifica la funzionalità di ricerca"""
        from src.rag_system import search_documents
        
        try:
            # Query di test
            test_query = "Quali sono i materiali plastici permessi per il contatto alimentare?"
            
            # Esegui ricerca
            results = search_documents(
                self.clients, 
                test_query,
                self.config.PINECONE_INDEX_NAME,
                top_k=3
            )
            
            # Verifica risultati
            self.assertIsNotNone(results)
            self.assertIsInstance(results, list)
            if len(results) > 0:
                print(f"✅ Ricerca RAG: trovati {len(results)} risultati")
                for i, result in enumerate(results):
                    print(f"   Risultato {i+1}: Score {result.score:.4f}")
                    print(f"   Fonte: {result.source}")
                    if result.page_number:
                        print(f"   Pagina: {result.page_number}")
            else:
                print("⚠️ Ricerca RAG: nessun risultato trovato (possibilmente normale)")
                
        except Exception as e:
            self.fail(f"❌ Ricerca RAG: errore - {e}")
    
    def test_response_generation(self):
        """Verifica la generazione di risposte"""
        from src.rag_system import search_documents, generate_rag_response
        
        try:
            # Query di test
            test_query = "Quali sono i requisiti per i materiali MOCA?"
            
            # Esegui ricerca
            results = search_documents(
                self.clients, 
                test_query,
                self.config.PINECONE_INDEX_NAME,
                top_k=3
            )
            
            # Genera risposta
            response, sources = generate_rag_response(
                self.clients,
                test_query,
                results,
                self.config.AZURE_CHAT_DEPLOYMENT
            )
            
            # Verifica risposta
            self.assertIsNotNone(response)
            self.assertTrue(len(response) > 0)
            print(f"✅ Generazione risposta: {len(response)} caratteri generati")
            print(f"   Fonti citate: {len(sources)}")
                
        except Exception as e:
            self.fail(f"❌ Generazione risposta: errore - {e}")

if __name__ == "__main__":
    print("🧪 Avvio test del sistema...")
    unittest.main()
