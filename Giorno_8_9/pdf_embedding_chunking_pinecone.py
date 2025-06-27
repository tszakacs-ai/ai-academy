import os
import json
import pickle
from typing import List, Dict, Any
from dotenv import load_dotenv
import numpy as np
from openai import AzureOpenAI
import re
from dataclasses import dataclass
from datetime import datetime
from pinecone import Pinecone

# Gestione import PyMuPDF con conflitti
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        # Fallback usando PyPDF2 se PyMuPDF non funziona
        try:
            import PyPDF2
            PYMUPDF_AVAILABLE = False
        except ImportError:
            raise ImportError("Nessuna libreria PDF disponibile. Installa PyMuPDF o PyPDF2")
    else:
        PYMUPDF_AVAILABLE = True
else:
    PYMUPDF_AVAILABLE = True

# Carica le variabili d'ambiente
load_dotenv()

@dataclass
class DocumentChunk:
    """Rappresenta un chunk del documento con i suoi metadati"""
    id: str
    content: str
    page_number: int
    chunk_index: int
    section: str
    subsection: str
    document_type: str
    embedding: List[float] = None
    metadata: Dict[str, Any] = None

class PDFPineconeProcessor:
    def __init__(self):
        """Inizializza il processore con i client Azure OpenAI e Pinecone"""
        # Debug: stampa le variabili caricate
        print("Caricamento variabili d'ambiente...")
        
        # Azure OpenAI per embeddings
        self.endpoint = os.getenv("ADA_ENDPOINT")
        print(f"ADA_ENDPOINT: {self.endpoint}")
        if not self.endpoint:
            raise ValueError("Devi definire ADA_ENDPOINT nel file .env")
        
        self.api_key = os.getenv("ADA_API_KEY")
        print(f"ADA_API_KEY: {'*' * len(self.api_key) if self.api_key else 'None'}")
        if not self.api_key:
            raise ValueError("Devi definire ADA_API_KEY nel file .env")
        
        self.api_version = os.getenv("ADA_API_VERSION", "2024-02-01")
        self.deployment_name = os.getenv("ADA_DEPLOYMENT_NAME", "text-embedding-ada-002")
        
        # Client Azure OpenAI
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # Pinecone setup
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Devi definire PINECONE_API_KEY nel file .env")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index("compliance50")
        
        # Configurazione chunking
        self.max_chunk_size = 1500  # Caratteri per chunk
        self.overlap_size = 200     # Overlap tra chunks
        
        # Pattern per identificare sezioni nel documento normativo
        self.section_patterns = {
            'regulation': r'Reg\.to \(.*?\)',
            'directive': r'Direttiva \d+/\d+/[A-Z]+',
            'article': r'art\.\s*\d+|articolo\s*\d+',
            'decree': r'Decreto.*?\d{4}',
            'chapter': r'Capo\s+[IVX]+|Titolo\s+[IVX]+',
            'definition': r'Definizioni:|definizioni:',
            'requirements': r'Requisiti|Obblighi',
            'procedure': r'Procedur[ae]|Modalità'
        }

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Estrae il testo dal PDF mantenendo informazioni sulla pagina"""
        if PYMUPDF_AVAILABLE:
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Estrae testo usando PyMuPDF"""
        pages_content = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Solo pagine con contenuto
                    pages_content.append({
                        'page_number': page_num + 1,
                        'content': text,
                        'char_count': len(text)
                    })
            
            doc.close()
            print(f"Estratto testo da {len(pages_content)} pagine con PyMuPDF")
            return pages_content
            
        except Exception as e:
            print(f"Errore nell'estrazione del PDF con PyMuPDF: {e}")
            return []
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Estrae testo usando PyPDF2 come fallback"""
        pages_content = []
        
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    if text.strip():  # Solo pagine con contenuto
                        pages_content.append({
                            'page_number': page_num + 1,
                            'content': text,
                            'char_count': len(text)
                        })
            
            print(f"Estratto testo da {len(pages_content)} pagine con PyPDF2")
            return pages_content
            
        except Exception as e:
            print(f"Errore nell'estrazione del PDF con PyPDF2: {e}")
            return []

    def identify_section(self, text: str) -> tuple:
        """Identifica la sezione e sottosezione del testo"""
        text_lower = text.lower()
        
        # Identifica il tipo di sezione principale
        section = "general"
        for section_type, pattern in self.section_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                section = section_type
                break
        
        # Estrae sottosezione specifica
        subsection = ""
        
        # Per regolamenti e direttive
        reg_match = re.search(r'(Reg\.to.*?\d{4}/\d+|Direttiva.*?\d{4}/\d+)', text, re.IGNORECASE)
        if reg_match:
            subsection = reg_match.group(1)
        
        # Per articoli
        art_match = re.search(r'(art\.\s*\d+[a-z]*|articolo\s*\d+[a-z]*)', text, re.IGNORECASE)
        if art_match:
            subsection = art_match.group(1)
        
        # Per definizioni specifiche
        if 'definizioni' in text_lower:
            def_match = re.search(r'([a-zA-Z\s]+):', text)
            if def_match:
                subsection = f"def_{def_match.group(1).strip()}"
        
        return section, subsection

    def smart_chunk_text(self, text: str, page_number: int) -> List[str]:
        """Divide il testo in chunks intelligenti basati sulla struttura del documento"""
        chunks = []
        
        # Dividi per paragrafi principali
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Se il paragrafo da solo è troppo lungo, dividilo
            if len(paragraph) > self.max_chunk_size:
                # Salva il chunk corrente se non vuoto
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Dividi il paragrafo lungo per frasi
                sentences = re.split(r'[.!?]+', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(temp_chunk + sentence) > self.max_chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        temp_chunk += sentence + ". "
                
                if temp_chunk:
                    current_chunk = temp_chunk
            
            # Se aggiungere questo paragrafo supera la dimensione massima
            elif len(current_chunk + paragraph) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Aggiungi l'ultimo chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def create_chunks(self, pages_content: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Crea chunks dal contenuto delle pagine"""
        all_chunks = []
        chunk_id_counter = 0
        
        for page_data in pages_content:
            page_number = page_data['page_number']
            content = page_data['content']
            
            # Dividi il contenuto della pagina in chunks
            text_chunks = self.smart_chunk_text(content, page_number)
            
            for chunk_index, chunk_text in enumerate(text_chunks):
                if len(chunk_text.strip()) < 50:  # Salta chunks troppo piccoli
                    continue
                
                # Identifica sezione e sottosezione
                section, subsection = self.identify_section(chunk_text)
                
                # Determina il tipo di documento
                doc_type = "normativa"
                if any(keyword in chunk_text.lower() for keyword in ['regolamento', 'reg.to']):
                    doc_type = "regolamento"
                elif any(keyword in chunk_text.lower() for keyword in ['direttiva']):
                    doc_type = "direttiva"
                elif any(keyword in chunk_text.lower() for keyword in ['decreto']):
                    doc_type = "decreto"
                
                # Crea metadata
                metadata = {
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'contains_definitions': 'definizioni' in chunk_text.lower(),
                    'contains_requirements': any(req in chunk_text.lower() for req in ['requisiti', 'obblighi', 'deve', 'devono']),
                    'contains_procedures': any(proc in chunk_text.lower() for proc in ['procedura', 'modalità', 'metodo']),
                    'extraction_timestamp': datetime.now().isoformat()
                }
                
                chunk = DocumentChunk(
                    id=f"chunk_{chunk_id_counter:06d}",
                    content=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    section=section,
                    subsection=subsection,
                    document_type=doc_type,
                    metadata=metadata
                )
                
                all_chunks.append(chunk)
                chunk_id_counter += 1
        
        print(f"Creati {len(all_chunks)} chunks dal documento")
        return all_chunks

    def get_embedding(self, text: str) -> List[float]:
        """Ottiene l'embedding per il testo usando Azure OpenAI"""
        try:
            # Pulisce il testo
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            
            response = self.client.embeddings.create(
                input=cleaned_text,
                model=self.deployment_name
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Errore nel creare embedding: {e}")
            return None

    def upload_to_pinecone(self, chunks: List[DocumentChunk], batch_size: int = 10):
        """Carica i chunks con embeddings su Pinecone"""
        print("Creazione embeddings e upload su Pinecone...")
        
        vectors_to_upsert = []
        
        for i, chunk in enumerate(chunks):
            # Crea embedding
            embedding = self.get_embedding(chunk.content)
            if not embedding:
                print(f"Saltato chunk {chunk.id} - embedding fallito")
                continue
            
            # Prepara metadata per Pinecone (solo valori semplici)
            pinecone_metadata = {
                'content': chunk.content,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'section': chunk.section,
                'subsection': chunk.subsection,
                'document_type': chunk.document_type,
                'char_count': chunk.metadata['char_count'],
                'word_count': chunk.metadata['word_count'],
                'contains_definitions': chunk.metadata['contains_definitions'],
                'contains_requirements': chunk.metadata['contains_requirements'],
                'contains_procedures': chunk.metadata['contains_procedures'],
                'extraction_timestamp': chunk.metadata['extraction_timestamp']
            }
            
            # Aggiungi al batch
            vectors_to_upsert.append({
                'id': chunk.id,
                'values': embedding,
                'metadata': pinecone_metadata
            })
            
            # Upload batch quando raggiunge la dimensione desiderata
            if len(vectors_to_upsert) >= batch_size:
                try:
                    self.index.upsert(vectors=vectors_to_upsert)
                    print(f"Caricati {len(vectors_to_upsert)} chunks su Pinecone")
                    vectors_to_upsert = []
                except Exception as e:
                    print(f"Errore nell'upload su Pinecone: {e}")
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Processati {i + 1}/{len(chunks)} chunks ({(i + 1)/len(chunks)*100:.1f}%)")
        
        # Upload degli ultimi chunks rimasti
        if vectors_to_upsert:
            try:
                self.index.upsert(vectors=vectors_to_upsert)
                print(f"Caricati ultimi {len(vectors_to_upsert)} chunks su Pinecone")
            except Exception as e:
                print(f"Errore nell'upload finale su Pinecone: {e}")
        
        print(f"Upload completato per {len(chunks)} chunks")

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Cerca chunks simili alla query"""
        # Crea embedding per la query
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            print("Errore nella creazione dell'embedding per la query")
            return []
        
        try:
            # Effettua la ricerca
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results['matches']
        except Exception as e:
            print(f"Errore nella ricerca: {e}")
            return []

    def get_index_stats(self) -> Dict:
        """Ottiene statistiche sull'indice Pinecone"""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"Errore nel recupero delle statistiche: {e}")
            return {}

def main():
    """Funzione principale per processare il PDF e caricarlo su Pinecone"""
    # Inizializza il processore
    processor = PDFPineconeProcessor()
    
    # Percorso del PDF
    pdf_path = "Gruppo_8/pdf_documents/Wiki_Conformità_primi_50.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File PDF non trovato: {pdf_path}")
        return
    
    print(f"Processando PDF: {pdf_path}")
    
    # Step 1: Estrai testo dal PDF
    pages_content = processor.extract_text_from_pdf(pdf_path)
    if not pages_content:
        print("Nessun contenuto estratto dal PDF")
        return
    
    # Step 2: Crea chunks
    chunks = processor.create_chunks(pages_content)
    if not chunks:
        print("Nessun chunk creato")
        return
    
    # Step 3: Carica su Pinecone con embeddings
    processor.upload_to_pinecone(chunks)
    
    # Step 4: Verifica statistiche
    stats = processor.get_index_stats()
    print("\n=== STATISTICHE PINECONE ===")
    print(f"Statistiche indice: {stats}")
    
    # Step 5: Test di ricerca
    print("\n=== TEST DI RICERCA ===")
    test_query = "requisiti per materiali a contatto con alimenti"
    results = processor.search_similar(test_query, top_k=3)
    
    print(f"Risultati per query: '{test_query}'")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.4f}")
        print(f"   Pagina: {result['metadata']['page_number']}")
        print(f"   Sezione: {result['metadata']['section']}")
        print(f"   Contenuto: {result['metadata']['content'][:200]}...")

if __name__ == "__main__":
    main()