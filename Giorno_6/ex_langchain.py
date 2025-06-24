import os
import faiss
import numpy as np
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import PyPDF2
from typing import List

class SimpleRAG:
    def __init__(self):
        """Sistema RAG semplificato con FAISS"""
        load_dotenv()
        
        # Inizializza Azure AI Project Client
        self.project = AIProjectClient(
            endpoint=os.getenv("PROJECT_ENDPOINT"),
            credential=DefaultAzureCredential()
        )
        
        self.client = self.project.inference.get_azure_openai_client(
            api_version="2024-10-21"
        )
        
        # Storage semplice
        self.documents = []
        self.index = None
        
    def load_pdf(self, pdf_path: str):
        """Carica e processa PDF"""
        print(f"📄 Caricamento: {pdf_path}")
        
        # Estrai testo
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        # Dividi in chunks semplici
        chunk_size = 1000
        self.documents = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        self.documents = [doc.strip() for doc in self.documents if doc.strip()]
        
        print(f"✅ Creati {len(self.documents)} chunks")
        
        # Crea embeddings
        embeddings = []
        for i, doc in enumerate(self.documents):
            print(f"🧠 Embedding {i+1}/{len(self.documents)}")
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=doc
            )
            embeddings.append(response.data[0].embedding)
        
        # Crea indice FAISS
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product per cosine similarity
        
        # Normalizza per cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        print(f"✅ Indice FAISS creato con {self.index.ntotal} documenti")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Cerca documenti rilevanti"""
        # Crea embedding per la query
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = np.array([response.data[0].embedding]).astype('float32')
        
        # Normalizza
        faiss.normalize_L2(query_embedding)
        
        # Cerca
        scores, indices = self.index.search(query_embedding, top_k)
        
        return [self.documents[i] for i in indices[0]]
    
    def ask(self, question: str) -> str:
        """Fai una domanda"""
        print(f"\n❓ {question}")
        
        # Cerca documenti rilevanti
        relevant_docs = self.search(question)
        context = "\n\n".join(relevant_docs)
        
        # Genera risposta
        messages = [
            {"role": "system", "content": "Rispondi basandoti solo sul contesto fornito. Rispondi in italiano."},
            {"role": "user", "content": f"Contesto:\n{context}\n\nDomanda: {question}"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        print(f"🤖 {answer}")
        return answer
    
    def chat(self):
        """Sessione di chat interattiva"""
        print("\n💬 Inizia a fare domande! ('quit' per uscire)")
        
        while True:
            try:
                question = input("\n❓ ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Ciao!")
                    break
                
                if question:
                    self.ask(question)
                    
            except KeyboardInterrupt:
                print("\n👋 Ciao!")
                break


def main():
    """Avvia il sistema"""
    print("🚀 Sistema RAG Semplificato con FAISS")
    print("=" * 40)
    
    # Inizializza
    rag = SimpleRAG()
    
    # Trova PDF
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ Nessun PDF trovato nella directory corrente")
        return
    
    if len(pdf_files) == 1:
        pdf_path = pdf_files[0]
    else:
        print("📁 File PDF trovati:")
        for i, pdf in enumerate(pdf_files):
            print(f"  {i+1}. {pdf}")
        
        choice = int(input("Scegli (numero): ")) - 1
        pdf_path = pdf_files[choice]
    
    # Carica e avvia chat
    rag.load_pdf(pdf_path)
    rag.chat()


if __name__ == "__main__":
    main()