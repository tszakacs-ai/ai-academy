import numpy as np
from typing import List, Dict, Tuple, Optional
import tiktoken
from openai import AzureOpenAI
import faiss
import pickle
from datetime import datetime
import json

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
        
        # initialize the FAISS index and document storage
        self.index = None
        self.documents = []
        self.embeddings = []
        
        # Tokenizer for counting tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # conversation history
        self.chat_history = []
        self.max_history_messages = 20  # max 10 messages in history
        self.max_history_tokens = 4000  # max tokens in history
        
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
    
    def generate_response(self, query: str, top_k: int = 3, max_tokens: int = 2000, 
                         use_chat_history: bool = True) -> Dict:
        """
        Genera una risposta usando RAG con memoria conversazionale
        
        Args:
            query: Domanda dell'utente
            top_k: Numero di documenti da utilizzare come contesto
            max_tokens: Numero massimo di token per la risposta
            use_chat_history: Se utilizzare la memoria conversazionale
            
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
        
        # Prepara i messaggi per la conversazione
        messages = self._prepare_messages_for_chat(query, context, use_chat_history)
        
        # Genera la risposta con GPT-4
        response = self.client.chat.completions.create(
            model=self.gpt_deployment,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        
        # Salva la conversazione nella storia
        if use_chat_history:
            self._add_to_chat_history(query, assistant_response, sources)
        
        return {
            'response': assistant_response,
            'sources': sources,
            'tokens_used': response.usage.total_tokens,
            'chat_history_length': len(self.chat_history)
        }
        
    def _prepare_messages_for_chat(self, query: str, context: str, use_chat_history: bool) -> List[Dict]:
        """Prepara i messaggi per l'API includendo la storia della chat"""
        
        # Messaggio di sistema aggiornato per includere la memoria
        system_prompt = """Sei un assistente esperto che risponde alle domande basandoti sul contesto fornito
        e sulla conversazione precedente. Segui queste regole:
        
        1. Usa principalmente il contesto fornito per rispondere alla domanda attuale
        2. Puoi fare riferimento a informazioni discusse precedentemente nella conversazione
        3. Se l'informazione non è nel contesto né nella conversazione precedente, dichiaralo esplicitamente
        4. Cita sempre da quale documento proviene l'informazione quando possibile
        5. Mantieni coerenza con le risposte precedenti nella conversazione
        6. Se la domanda fa riferimento a "quello di cui abbiamo parlato prima" o simili, usa la conversazione precedente"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Aggiungi la storia della chat se richiesta e disponibile
        if use_chat_history and self.chat_history:
            history_messages = self._get_relevant_chat_history()
            messages.extend(history_messages)
        
        # Aggiungi il contesto attuale e la domanda
        user_content = f"""Contesto attuale:
                        {context}

                        Domanda: {query}

                        Rispondi basandoti sul contesto fornito e sulla conversazione precedente se pertinente.
                        Se citi informazioni, indica da quale documento provengono."""
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def _get_relevant_chat_history(self) -> List[Dict]:
        """Restituisce la storia della chat che rientra nel limite di token"""
        if not self.chat_history:
            return []
        
        history_messages = []
        current_tokens = 0
        
        # Parti dai messaggi più recenti
        for message in reversed(self.chat_history):
            message_tokens = len(self.encoding.encode(message['content']))
            
            if current_tokens + message_tokens > self.max_history_tokens:
                break
                
            history_messages.insert(0, {
                "role": message['role'],
                "content": message['content']
            })
            current_tokens += message_tokens
        
        return history_messages
    
    def _add_to_chat_history(self, user_query: str, assistant_response: str, sources: List[Dict]):
        """Aggiunge la conversazione alla storia"""
        timestamp = datetime.now().isoformat()
        
        # Aggiungi il messaggio dell'utente
        self.chat_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': timestamp
        })
        
        # Aggiungi la risposta dell'assistente con info sulle fonti
        assistant_content = assistant_response
        if sources:
            sources_info = "\n[Fonti consultate: " + ", ".join([s['title'] for s in sources]) + "]"
            assistant_content += sources_info
        
        self.chat_history.append({
            'role': 'assistant',
            'content': assistant_content,
            'sources': sources,
            'timestamp': timestamp
        })
        
        # Mantieni solo gli ultimi N messaggi
        if len(self.chat_history) > self.max_history_messages:
            self.chat_history = self.chat_history[-self.max_history_messages:]
    
    def clear_chat_history(self):
        """Cancella la storia della conversazione"""
        self.chat_history.clear()
        print("Storia della conversazione cancellata.")
    
    def get_chat_summary(self) -> Dict:
        """Restituisce un riassunto della conversazione corrente"""
        if not self.chat_history:
            return {"message": "Nessuna conversazione in corso"}
        
        user_messages = [msg for msg in self.chat_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in self.chat_history if msg['role'] == 'assistant']
        
        return {
            "total_exchanges": len(user_messages),
            "total_messages": len(self.chat_history),
            "first_message_time": self.chat_history[0]['timestamp'] if self.chat_history else None,
            "last_message_time": self.chat_history[-1]['timestamp'] if self.chat_history else None,
            "recent_topics": self._extract_recent_topics()
        }
    
    def _extract_recent_topics(self) -> List[str]:
        """Estrae gli argomenti recenti dalla conversazione"""
        topics = []
        user_messages = [msg for msg in self.chat_history if msg['role'] == 'user']
        
        # Prendi le ultime 5 domande dell'utente
        for message in user_messages[-5:]:
            # Prendi le prime parole significative
            words = message['content'].split()[:6]
            topic = " ".join(words)
            if topic not in topics:
                topics.append(topic)
        
        return topics
    
    def save_chat_history(self, filepath: str):
        """Salva la storia della conversazione su file"""
        chat_data = {
            "chat_history": self.chat_history,
            "summary": self.get_chat_summary(),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
        print(f"Storia della conversazione salvata in: {filepath}")
    
    def load_chat_history(self, filepath: str):
        """Carica una storia della conversazione da file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            
            self.chat_history = chat_data.get("chat_history", [])
            print(f"Storia della conversazione caricata da: {filepath}")
            print(f"Messaggi caricati: {len(self.chat_history)}")
            
        except FileNotFoundError:
            print(f"File non trovato: {filepath}")
        except json.JSONDecodeError:
            print(f"Errore nel parsing del file JSON: {filepath}")
    
    def continue_conversation(self, query: str, top_k: int = 3, max_tokens: int = 2000) -> Dict:
        """
        Metodo di convenienza per continuare una conversazione
        (equivale a generate_response con use_chat_history=True)
        """
        return self.generate_response(query, top_k, max_tokens, use_chat_history=True)
    
    def start_new_conversation(self, query: str, top_k: int = 3, max_tokens: int = 2000) -> Dict:
        """
        Metodo di convenienza per iniziare una nuova conversazione
        (cancella la storia e genera una risposta)
        """
        self.clear_chat_history()
        return self.generate_response(query, top_k, max_tokens, use_chat_history=False)
