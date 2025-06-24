import os
import numpy as np
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from typing import List, Dict, Any
import sys

class InteractiveRAGSystem:
    def __init__(self, project_endpoint: str = None, model_name: str = "gpt-4o", embedding_model: str = "text-embedding-ada-002"):
        """Sistema RAG interattivo senza LangChain"""
        
        load_dotenv()
        
        if project_endpoint is None:
            project_endpoint = os.getenv("PROJECT_ENDPOINT")
        
        self.project_endpoint = project_endpoint
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # Inizializza Azure AI Project Client
        self.project = AIProjectClient(
            endpoint=project_endpoint,
            credential=DefaultAzureCredential()
        )
        
        # Client OpenAI tramite Azure AI Foundry
        self.azure_openai_client = self.project.inference.get_azure_openai_client(
            api_version="2024-10-21"
        )
        
        # Storage per documenti e embeddings
        self.documents = []
        self.embeddings = []
        self.document_embeddings = None
        self.document_loaded = False
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Estrae il testo da un file PDF"""
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Il file {pdf_path} non esiste.")
        
        print(f"📄 Estrazione testo da: {pdf_path}")
        
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"📖 Trovate {len(pdf_reader.pages)} pagine")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Pagina {page_num + 1} ---\n{page_text}\n"
                    print(f"   ✓ Pagina {page_num + 1} processata")
                    
        except Exception as e:
            print(f"❌ Errore nell'estrazione del PDF: {e}")
            raise
        
        print(f"✅ Testo estratto: {len(text)} caratteri")
        return text
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Divide il testo in chunks con sovrapposizione"""
        
        print("✂️  Divisione del testo in chunks...")
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Trova la fine della frase più vicina per evitare di tagliare a metà
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_space = chunk.rfind(' ')
                
                # Usa il separatore più vicino alla fine
                split_point = max(last_period, last_newline, last_space)
                if split_point > start + chunk_size // 2:  # Assicurati che il chunk non sia troppo piccolo
                    chunk = text[start:start + split_point + 1]
                    end = start + split_point + 1
            
            if chunk.strip():  # Solo se il chunk non è vuoto
                chunks.append(chunk.strip())
            
            start = end - overlap
            
            if start >= len(text):
                break
        
        print(f"✅ Testo diviso in {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Crea embeddings per una lista di testi"""
        
        print(f"🧠 Creazione embeddings per {len(texts)} testi...")
        
        try:
            # Processa in batch per evitare limiti di API
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"   📊 Processando batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                response = self.azure_openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)
            
            print("✅ Embeddings creati con successo!")
            return all_embeddings
            
        except Exception as e:
            print(f"❌ Errore nella creazione embeddings: {e}")
            raise
    
    def load_and_index_document(self, pdf_path: str):
        """Carica, processa e indicizza un documento PDF"""
        
        print("\n" + "="*60)
        print("🚀 CARICAMENTO E INDICIZZAZIONE DOCUMENTO")
        print("="*60)
        
        # Estrai testo dal PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Dividi in chunks
        self.documents = self.split_text_into_chunks(text)
        
        if not self.documents:
            raise ValueError("Nessun contenuto estratto dal PDF")
        
        # Crea embeddings
        self.embeddings = self.create_embeddings(self.documents)
        self.document_embeddings = np.array(self.embeddings)
        
        self.document_loaded = True
        
        print(f"✅ Documento {pdf_path} indicizzato con successo!")
        print(f"   📚 {len(self.documents)} chunks creati")
        print(f"   🧠 {len(self.embeddings)} embeddings generati")
        print(f"   📏 Dimensioni embedding: {len(self.embeddings[0])}")
    
    def find_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Trova i documenti più rilevanti per una query"""
        
        if not self.document_loaded:
            raise ValueError("❌ Nessun documento caricato. Esegui prima load_and_index_document()")
        
        # Crea embedding per la query
        query_embedding = self.create_embeddings([query])[0]
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Calcola similarità con tutti i documenti
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Trova i top_k documenti più simili
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_docs = []
        for i in top_indices:
            relevant_docs.append({
                'content': self.documents[i],
                'similarity': similarities[i],
                'index': i
            })
        
        return relevant_docs
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Genera una risposta basata sui documenti rilevanti"""
        
        # Prepara il contesto
        context = "\n\n---\n\n".join([doc['content'] for doc in context_docs])
        
        # Template del prompt
        system_prompt = """Sei un assistente AI esperto che risponde a domande basandoti sul contesto fornito dal documento.

ISTRUZIONI:
1. Usa SOLO le informazioni presenti nel contesto per rispondere
2. Se la domanda non può essere risposta con il contesto, dillo chiaramente
3. Sii preciso e dettagliato quando possibile
4. Rispondi in italiano in modo naturale e professionale
5. Se nel contesto ci sono informazioni parziali, specificalo

Rispondi sempre in modo utile e completo."""
        
        user_prompt = f"""CONTESTO DAL DOCUMENTO:
{context}

DOMANDA DELL'UTENTE: {query}

RISPOSTA:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.azure_openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1500,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Errore nella generazione della risposta: {e}"
    
    def ask_question(self, question: str, show_sources: bool = True) -> Dict[str, Any]:
        """Pone una domanda al sistema RAG"""
        
        print(f"\n🤔 Domanda: {question}")
        print("-" * 60)
        
        try:
            # Trova documenti rilevanti
            relevant_docs = self.find_relevant_documents(question)
            
            # Genera risposta
            answer = self.generate_answer(question, relevant_docs)
            
            print(f"🤖 Risposta:\n{answer}")
            
            if show_sources:
                print(f"\n📚 Fonti utilizzate ({len(relevant_docs)} documenti):")
                for i, doc in enumerate(relevant_docs, 1):
                    similarity_percent = doc['similarity'] * 100
                    print(f"\n   📄 Fonte {i} (Rilevanza: {similarity_percent:.1f}%):")
                    preview = doc['content'][:200].replace('\n', ' ')
                    print(f"      {preview}{'...' if len(doc['content']) > 200 else ''}")
            
            return {
                'question': question,
                'answer': answer,
                'sources': relevant_docs,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"❌ Errore nel processare la domanda: {e}"
            print(error_msg)
            return {
                'question': question,
                'answer': error_msg,
                'sources': [],
                'success': False
            }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche sui documenti caricati"""
        
        if not self.document_loaded:
            return {"message": "❌ Nessun documento caricato"}
        
        chunk_lengths = [len(doc) for doc in self.documents]
        
        return {
            "total_chunks": len(self.documents),
            "total_characters": sum(chunk_lengths),
            "average_chunk_length": int(np.mean(chunk_lengths)),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "embedding_dimensions": len(self.embeddings[0]) if self.embeddings else 0
        }
    
    def interactive_session(self):
        """Avvia una sessione interattiva di domande e risposte"""
        
        print("\n" + "="*60)
        print("💬 SESSIONE INTERATTIVA - FAI LE TUE DOMANDE!")
        print("="*60)
        print("ℹ️  Comandi disponibili:")
        print("   • Scrivi la tua domanda e premi INVIO")
        print("   • 'stats' - Mostra statistiche del documento")
        print("   • 'help' - Mostra questo aiuto")
        print("   • 'quit' o 'exit' - Esci dal programma")
        print("   • 'clear' - Pulisci lo schermo")
        print("-" * 60)
        
        if not self.document_loaded:
            print("⚠️  ATTENZIONE: Nessun documento caricato!")
            return
        
        question_count = 0
        
        while True:
            try:
                print(f"\n📝 Domanda #{question_count + 1}:")
                user_input = input("❓ ").strip()
                
                if not user_input:
                    print("⚠️  Inserisci una domanda valida!")
                    continue
                
                # Comandi speciali
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Grazie per aver usato il sistema RAG!")
                    print("🔚 Sessione terminata.")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n📖 COMANDI DISPONIBILI:")
                    print("   • Scrivi qualsiasi domanda sul documento")
                    print("   • 'stats' - Statistiche del documento")
                    print("   • 'clear' - Pulisci lo schermo")
                    print("   • 'quit' - Esci dal programma")
                    continue
                
                elif user_input.lower() == 'stats':
                    stats = self.get_document_stats()
                    print("\n📊 STATISTICHE DOCUMENTO:")
                    for key, value in stats.items():
                        print(f"   • {key.replace('_', ' ').title()}: {value}")
                    continue
                
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("🚀 Sistema RAG - Sessione Interattiva")
                    print("="*60)
                    continue
                
                # Processa la domanda
                question_count += 1
                result = self.ask_question(user_input)
                
                if result['success']:
                    print(f"\n✅ Domanda processata con successo!")
                else:
                    print(f"\n❌ Errore nel processare la domanda.")
                
            except KeyboardInterrupt:
                print("\n\n🛑 Interruzione da tastiera (Ctrl+C)")
                print("👋 Sessione terminata.")
                break
            except EOFError:
                print("\n\n🔚 Fine input.")
                print("👋 Sessione terminata.")
                break
            except Exception as e:
                print(f"\n❌ Errore imprevisto: {e}")
                continue


def find_pdf_file() -> str:
    """Trova il file PDF in diversi percorsi possibili"""
    
    possible_paths = [
        "Dubai Brochure.pdf",
        "Giorno_6/Dubai Brochure.pdf", 
        "ai-academy-1/Giorno_6/Dubai Brochure.pdf",
        "../Dubai Brochure.pdf",
        "../../Dubai Brochure.pdf",
        os.path.join(os.getcwd(), "Dubai Brochure.pdf"),
        os.path.join(os.path.dirname(__file__), "Dubai Brochure.pdf")
    ]
    
    print("🔍 Ricerca automatica del file PDF...")
    for i, path in enumerate(possible_paths, 1):
        full_path = os.path.abspath(path)
        print(f"   {i}. {full_path}")
        if os.path.exists(path):
            print(f"   ✅ TROVATO!")
            return path
    
    print("\n❌ File PDF non trovato automaticamente.")
    
    # Mostra i file PDF nella directory corrente
    current_dir = os.getcwd()
    pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf')]
    
    if pdf_files:
        print(f"\n📁 File PDF trovati nella directory corrente ({current_dir}):")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"   {i}. {pdf_file}")
        
        print(f"\n💡 Vuoi usare uno di questi file? (1-{len(pdf_files)}) o inserisci il percorso completo:")
        user_choice = input("Scelta: ").strip()
        
        # Controlla se è un numero
        try:
            choice_num = int(user_choice)
            if 1 <= choice_num <= len(pdf_files):
                return pdf_files[choice_num - 1]
        except ValueError:
            pass
        
        # Altrimenti usa come percorso
        if os.path.exists(user_choice):
            return user_choice
    
    # Chiedi percorso manuale
    print(f"\n📝 Inserisci il percorso completo del file PDF:")
    user_path = input("Percorso: ").strip().strip('"').strip("'")
    
    if os.path.exists(user_path):
        return user_path
    else:
        raise FileNotFoundError(f"Il file {user_path} non esiste.")


def main():
    """Funzione principale per la sessione interattiva"""
    
    print("🚀 SISTEMA RAG INTERATTIVO")
    print("🔗 Powered by Azure AI Foundry")
    print("="*60)
    
    try:
        # Inizializza il sistema
        print("⚙️  Inizializzazione sistema...")
        rag_system = InteractiveRAGSystem()
        print("✅ Sistema inizializzato!")
        
        # Trova e carica il file PDF
        pdf_path = find_pdf_file()
        rag_system.load_and_index_document(pdf_path)
        
        # Mostra statistiche
        stats = rag_system.get_document_stats()
        print(f"\n📊 STATISTICHE DOCUMENTO:")
        for key, value in stats.items():
            if key != "message":
                print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        # Avvia sessione interattiva
        rag_system.interactive_session()
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRORE FILE: {e}")
        print("💡 Assicurati che il file PDF esista nel percorso specificato.")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        print("\n🔧 Dettagli tecnici:")
        traceback.print_exc()
        
    finally:
        print(f"\n👋 Arrivederci!")


if __name__ == "__main__":
    main()