"""
Sistema completo per interrogare documenti PDF usando Pinecone e Azure OpenAI
Utilizza le credenziali Azure OpenAI e l'indice Pinecone 'compdf' esistente
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Carica variabili d'ambiente
load_dotenv()

@dataclass
class QueryResult:
    """Rappresenta un risultato di query con metadati"""
    content: str
    score: float
    page_number: Optional[int] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = None

class PDFQuerySystem:
    """Sistema principale per interrogare documenti PDF"""
    
    def __init__(self):
        """Inizializza il sistema con le credenziali Azure OpenAI e Pinecone"""
        self._setup_azure_openai()
        self._setup_pinecone()
    
    def _setup_azure_openai(self):
        """Configura il client Azure OpenAI per embeddings e chat"""
        try:
            from openai import AzureOpenAI
            
            # Client per embeddings
            self.embedding_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
                api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
                api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
            )
            
            # Client per chat completions
            self.chat_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            )
            
            self.embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            self.chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            
            print("✅ Azure OpenAI configurato correttamente")
            
        except Exception as e:
            print(f"❌ Errore configurazione Azure OpenAI: {e}")
            raise
    
    def _setup_pinecone(self):
        """Configura la connessione a Pinecone"""
        try:
            from pinecone import Pinecone
            
            # Inizializza Pinecone con la nuova API
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            # Usa l'indice 'compdf' esistente
            self.index_name = "compdf"
            self.index = self.pc.Index(self.index_name)
            
            # Verifica statistiche dell'indice
            stats = self.index.describe_index_stats()
            print(f"✅ Connesso all'indice Pinecone '{self.index_name}'")
            print(f"   - Dimensioni: {stats.dimension}")
            print(f"   - Vettori totali: {stats.total_vector_count}")
            
        except Exception as e:
            print(f"❌ Errore configurazione Pinecone: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Genera embedding per il testo usando Azure OpenAI"""
        try:
            # Pulisce il testo
            cleaned_text = text.strip().replace('\n', ' ')
            
            response = self.embedding_client.embeddings.create(
                input=cleaned_text,
                model=self.embedding_deployment
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"❌ Errore generazione embedding: {e}")
            return None
    
    def search_documents(self, query: str, top_k: int = 5, score_threshold: float = 0.7) -> List[QueryResult]:
        """Cerca documenti simili alla query"""
        try:
            # Genera embedding per la query
            print(f"🔍 Ricerca: '{query}'")
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return []
            
            # Esegui ricerca in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Converti risultati
            results = []
            for match in search_results.matches:
                if match.score >= score_threshold:
                    metadata = match.metadata or {}
                    
                    # Gestisci diverse possibili chiavi per il contenuto
                    content = metadata.get('text', metadata.get('content', 'Contenuto non disponibile'))
                    page_number = metadata.get('page_number', metadata.get('page'))
                    
                    result = QueryResult(
                        content=str(content),
                        score=match.score,
                        page_number=page_number,
                        source=metadata.get('source', 'Sconosciuta'),
                        metadata=metadata
                    )
                    results.append(result)
            
            print(f"✅ Trovati {len(results)} risultati con score >= {score_threshold}")
            return results
            
        except Exception as e:
            print(f"❌ Errore ricerca: {e}")
            return []
    
    def generate_answer(self, query: str, search_results: List[QueryResult], max_context_length: int = 8000) -> str:
        """Genera una risposta usando i risultati della ricerca"""
        try:
            if not search_results:
                return "Non sono riuscito a trovare informazioni rilevanti per la tua domanda nei documenti."
            
            # Costruisci contesto dai risultati
            context_parts = []
            current_length = 0
            
            for result in search_results:
                content_snippet = result.content[:1000]  # Limita ogni snippet
                source_info = f"[Fonte: {result.source}, Pagina: {result.page_number}, Score: {result.score:.3f}]"
                
                part = f"{source_info}\n{content_snippet}\n"
                
                if current_length + len(part) > max_context_length:
                    break
                
                context_parts.append(part)
                current_length += len(part)
            
            context = "\n---\n".join(context_parts)
            
            # Template per il prompt
            system_prompt = """Sei un assistente esperto in normative e conformità per materiali a contatto con alimenti (MOCA).
Analizza attentamente i documenti forniti e rispondi in modo preciso e dettagliato.

ISTRUZIONI:
1. Basa la risposta SOLO sui documenti forniti
2. Cita sempre le fonti specifiche (pagina, documento)
3. Se l'informazione non è nei documenti, dichiaralo chiaramente
4. Mantieni un linguaggio tecnico ma comprensibile
5. Organizza la risposta in modo strutturato

FORMATO RISPOSTA:
- Risposta diretta alla domanda
- Riferimenti normativi specifici
- Eventuali considerazioni aggiuntive"""

            user_prompt = f"""DOMANDA: {query}

DOCUMENTI DI RIFERIMENTO:
{context}

Fornisci una risposta completa e precisa basata sui documenti sopra."""

            # Genera risposta
            response = self.chat_client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Errore generazione risposta: {e}")
            return f"Errore nella generazione della risposta: {e}"
    
    def query(self, question: str, top_k: int = 5, score_threshold: float = 0.7) -> Dict[str, Any]:
        """Esegue una query completa: ricerca + generazione risposta"""
        print(f"\n🚀 QUERY: {question}")
        print("=" * 80)
        
        # Step 1: Ricerca documenti
        search_results = self.search_documents(question, top_k, score_threshold)
        
        if not search_results:
            return {
                "question": question,
                "answer": "Non ho trovato informazioni rilevanti nei documenti per rispondere alla tua domanda.",
                "sources": [],
                "search_results": []
            }
        
        # Step 2: Genera risposta
        answer = self.generate_answer(question, search_results)
        
        # Step 3: Prepara risultato finale
        sources = []
        for result in search_results:
            sources.append({
                "source": result.source,
                "page": result.page_number,
                "score": round(result.score, 3),
                "preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
            })
        
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "search_results": search_results,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def interactive_mode(self):
        """Modalità interattiva per porre domande"""
        print("\n🤖 SISTEMA INTERATTIVO DI INTERROGAZIONE PDF")
        print("Digita 'quit' per uscire")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n❓ Inserisci la tua domanda: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Arrivederci!")
                    break
                
                if not question:
                    continue
                
                # Esegui query
                result = self.query(question)
                
                # Mostra risultati
                print(f"\n💡 RISPOSTA:")
                print("-" * 40)
                print(result['answer'])
                
                print(f"\n📚 FONTI CONSULTATE ({len(result['sources'])}):")
                print("-" * 40)
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['source']} (Pagina {source['page']}) - Score: {source['score']}")
                    print(f"   Preview: {source['preview']}")
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interruzione da tastiera. Arrivederci!")
                break
            except Exception as e:
                print(f"❌ Errore: {e}")

def main():
    """Funzione principale per testare il sistema"""
    try:
        # Inizializza sistema
        pdf_system = PDFQuerySystem()
        
        # Esempi di domande predefinite
        sample_questions = [
            "Quali sono i requisiti per i materiali a contatto con alimenti secondo il regolamento 1935/2004?",
            "Che cosa sono i materiali attivi e intelligenti?",
            "Quali sono i limiti di migrazione per le materie plastiche?",
            "Come deve essere effettuata l'etichettatura dei MOCA?",
            "Cosa prevede il decreto ministeriale 21 marzo 1973?"
        ]
        
        print("🔧 ESEMPI DI DOMANDE PREDEFINITE:")
        for i, q in enumerate(sample_questions, 1):
            print(f"{i}. {q}")
        
        choice = input(f"\nVuoi provare una domanda predefinita (1-{len(sample_questions)}) o modalità interattiva (i)? ").strip()
        
        if choice.lower() == 'i':
            # Modalità interattiva
            pdf_system.interactive_mode()
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(sample_questions):
                    question = sample_questions[idx]
                    result = pdf_system.query(question)
                    
                    print(f"\n💡 RISPOSTA:")
                    print("-" * 40)
                    print(result['answer'])
                    
                    print(f"\n📚 FONTI CONSULTATE:")
                    print("-" * 40)
                    for source in result['sources']:
                        print(f"• {source['source']} (Pag. {source['page']}) - Score: {source['score']}")
                else:
                    print("❌ Scelta non valida")
            except ValueError:
                print("❌ Input non valido")
        
    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")

if __name__ == "__main__":
    main()