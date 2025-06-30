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

class PDFEmbeddingProcessor:
    def __init__(self):
        """Inizializza il processore con il client Azure OpenAI"""
        # Debug: stampa le variabili caricate
        print("Caricamento variabili d'ambiente...")
        
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
        
        # Inizializza il client Azure OpenAI
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
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
        pages_content = []
        
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

    def create_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 10) -> List[DocumentChunk]:
        """Crea embeddings per tutti i chunks"""
        print("Creazione embeddings in corso...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                embedding = self.get_embedding(chunk.content)
                if embedding:
                    chunk.embedding = embedding
                else:
                    print(f"Fallito embedding per chunk {chunk.id}")
            
            # Progress update
            progress = min(i + batch_size, len(chunks))
            print(f"Processati {progress}/{len(chunks)} chunks ({progress/len(chunks)*100:.1f}%)")
        
        # Filtra chunks senza embedding
        chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding is not None]
        print(f"Completati embeddings per {len(chunks_with_embeddings)}/{len(chunks)} chunks")
        
        return chunks_with_embeddings

    def save_chunks(self, chunks: List[DocumentChunk], output_path: str):
        """Salva i chunks con embeddings"""
        # Converti in formato serializzabile
        chunks_data = []
        for chunk in chunks:
            chunk_dict = {
                'id': chunk.id,
                'content': chunk.content,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'section': chunk.section,
                'subsection': chunk.subsection,
                'document_type': chunk.document_type,
                'embedding': chunk.embedding,
                'metadata': chunk.metadata
            }
            chunks_data.append(chunk_dict)
        
        # Salva in formato pickle per mantenere gli embeddings
        with open(output_path, 'wb') as f:
            pickle.dump(chunks_data, f)
        
        # Salva anche un file JSON senza embeddings per ispezione
        json_path = output_path.replace('.pkl', '_metadata.json')
        json_data = []
        for chunk_dict in chunks_data:
            json_chunk = chunk_dict.copy()
            del json_chunk['embedding']  # Rimuovi embedding per JSON
            json_data.append(json_chunk)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Chunks salvati in:")
        print(f"  - {output_path} (con embeddings)")
        print(f"  - {json_path} (metadata)")

    def load_chunks(self, file_path: str) -> List[DocumentChunk]:
        """Carica i chunks da file"""
        with open(file_path, 'rb') as f:
            chunks_data = pickle.load(f)
        
        chunks = []
        for data in chunks_data:
            chunk = DocumentChunk(
                id=data['id'],
                content=data['content'],
                page_number=data['page_number'],
                chunk_index=data['chunk_index'],
                section=data['section'],
                subsection=data['subsection'],
                document_type=data['document_type'],
                embedding=data['embedding'],
                metadata=data['metadata']
            )
            chunks.append(chunk)
        
        return chunks

    def get_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Restituisce statistiche sui chunks"""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'total_pages': len(set(chunk.page_number for chunk in chunks)),
            'avg_chunk_size': np.mean([len(chunk.content) for chunk in chunks]),
            'median_chunk_size': np.median([len(chunk.content) for chunk in chunks]),
            'sections': {},
            'document_types': {},
            'chunks_with_embeddings': sum(1 for chunk in chunks if chunk.embedding is not None)
        }
        
        # Conta per sezioni
        for chunk in chunks:
            stats['sections'][chunk.section] = stats['sections'].get(chunk.section, 0) + 1
            stats['document_types'][chunk.document_type] = stats['document_types'].get(chunk.document_type, 0) + 1
        
        return stats

def main():
    """Funzione principale per processare il PDF"""
    # Inizializza il processore
    processor = PDFEmbeddingProcessor()
    
    # Percorsi file
    pdf_path = "Wiki_Conformità_primi_50.pdf"
    output_path = "chunks_with_embeddings.pkl"
    
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
    
    # Step 3: Crea embeddings
    chunks_with_embeddings = processor.create_embeddings(chunks)
    
    # Step 4: Salva risultati
    processor.save_chunks(chunks_with_embeddings, output_path)
    
    # Step 5: Mostra statistiche
    stats = processor.get_stats(chunks_with_embeddings)
    print("\n=== STATISTICHE FINALI ===")
    print(f"Chunks totali: {stats['total_chunks']}")
    print(f"Pagine processate: {stats['total_pages']}")
    print(f"Dimensione media chunk: {stats['avg_chunk_size']:.0f} caratteri")
    print(f"Chunks con embeddings: {stats['chunks_with_embeddings']}")
    print(f"Sezioni identificate: {list(stats['sections'].keys())}")
    print(f"Tipi documento: {list(stats['document_types'].keys())}")

if __name__ == "__main__":
    main()