import numpy as np
from typing import List, Dict, Tuple
import tiktoken
from openai import AzureOpenAI
import faiss
import pickle
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

class AzureRAGSystem:
    """Sistema RAG utilizzando Azure OpenAI per embeddings e generazione"""
    
    def __init__(self, 
                 azure_endpoint: str,
                 api_key: str,
                 embedding_deployment: str = "text-embedding-ada-002",
                 gpt_deployment: str = "gpt-4",
                 api_version: str = "2024-02-01"):
        """
        Inizializza il sistema RAG con Azure OpenAI
        
        Args:
            azure_endpoint: L'endpoint del tuo servizio Azure OpenAI
            api_key: La chiave API per Azure OpenAI
            embedding_deployment: Nome del deployment per il modello di embedding
            gpt_deployment: Nome del deployment per GPT-4
            api_version: Versione API di Azure OpenAI
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        self.embedding_deployment = embedding_deployment
        self.gpt_deployment = gpt_deployment
        
        # Inizializza l'indice FAISS per la ricerca vettoriale
        self.index = None
        self.documents = []
        self.embeddings = []
        
        # Tokenizer per contare i token
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Divide il testo in chunk con overlap per mantenere il contesto
        
        Args:
            text: Testo da dividere
            chunk_size: Dimensione massima di ogni chunk in caratteri
            overlap: Numero di caratteri di sovrapposizione tra chunk
        
        Returns:
            Lista di chunk di testo
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Trova un punto di interruzione naturale (fine frase)
            if end < len(text):
                # Cerca il punto più vicino
                for sep in ['. ', '.\n', '? ', '! ', '\n\n']:
                    sep_pos = text.rfind(sep, start, end)
                    if sep_pos != -1:
                        end = sep_pos + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Ottiene l'embedding per un testo usando Azure OpenAI
        
        Args:
            text: Testo da convertire in embedding
            
        Returns:
            Vettore di embedding
        """
        response = self.client.embeddings.create(
            model=self.embedding_deployment,
            input=text
        )
        return response.data[0].embedding
    
    def add_documents(self, documents: List[Dict[str, str]], chunk_size: int = 1000):
        """
        Aggiunge documenti al corpus RAG
        
        Args:
            documents: Lista di documenti con 'title' e 'content'
            chunk_size: Dimensione dei chunk per documenti lunghi
        """
        all_chunks = []
        
        for doc in documents:
            # Chunking del documento
            chunks = self.chunk_text(doc['content'], chunk_size)
            
            # Aggiungi metadati a ogni chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'title': doc['title'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'metadata': doc.get('metadata', {})
                })
        
        # Genera embeddings per tutti i chunk
        print(f"Generazione embeddings per {len(all_chunks)} chunk...")
        embeddings = []
        
        for i, chunk in enumerate(all_chunks):
            print(f"Processati {i}/{len(all_chunks)} chunk")         
            embedding = self.get_embedding(chunk['text'])
            embeddings.append(embedding)
        
        # Aggiorna l'indice FAISS
        embeddings_array = np.array(embeddings).astype('float32')
        
        if self.index is None:
            # Crea nuovo indice
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product per cosine similarity
            faiss.normalize_L2(embeddings_array)  # Normalizza per cosine similarity
        
        self.index.add(embeddings_array)
        self.documents.extend(all_chunks)
        self.embeddings.extend(embeddings)
        
        print(f"Aggiunti {len(all_chunks)} chunk all'indice. Totale documenti: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Cerca i documenti più rilevanti per la query
        
        Args:
            query: Query di ricerca
            top_k: Numero di risultati da restituire
            
        Returns:
            Lista di tuple (documento, score)
        """
        if self.index is None or len(self.documents) == 0:
            raise ValueError("Nessun documento nel corpus. Usa add_documents() prima.")
        
        # Ottieni embedding della query
        query_embedding = np.array([self.get_embedding(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Cerca i documenti più simili
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def generate_response(self, query: str, top_k: int = 3, max_tokens: int = 2000) -> Dict:
        """
        Genera una risposta usando RAG
        
        Args:
            query: Domanda dell'utente
            top_k: Numero di documenti da utilizzare come contesto
            max_tokens: Numero massimo di token per la risposta
            
        Returns:
            Dizionario con risposta e metadati
        """
        # Recupera documenti rilevanti
        relevant_docs = self.search(query, top_k)
        
        # Costruisci il contesto dai documenti recuperati
        context_parts = []
        sources = []
        
        for doc, score in relevant_docs:
            context_parts.append(f"[Documento: {doc['title']}]\n{doc['text']}")
            sources.append({
                'title': doc['title'],
                'score': score,
                'chunk_index': doc['chunk_index']
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Conta i token per assicurarsi di non superare il limite
        context_tokens = len(self.encoding.encode(context))
        query_tokens = len(self.encoding.encode(query))
        
        # Lascia spazio per la risposta
        max_context_tokens = 6000  # Limite conservativo per GPT-4
        if context_tokens + query_tokens > max_context_tokens:
            # Tronca il contesto se necessario
            context = self.encoding.decode(
                self.encoding.encode(context)[:max_context_tokens - query_tokens]
            )
        
        # Costruisci il prompt
        system_prompt = """Sei un assistente esperto che risponde alle domande basandoti esclusivamente 
        sul contesto fornito. Se l'informazione richiesta non è presente nel contesto, 
        dichiaralo esplicitamente. Cita sempre da quale documento proviene l'informazione."""
        
        user_prompt = f"""Contesto:
        {context}

        Domanda: {query}

        Rispondi alla domanda basandoti esclusivamente sul contesto fornito sopra. 
        Se citi informazioni, indica da quale documento provengono."""
        
        # Genera la risposta con GPT-4
        response = self.client.chat.completions.create(
            model=self.gpt_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return {
            'response': response.choices[0].message.content,
            'sources': sources,
            'tokens_used': response.usage.total_tokens
        }
