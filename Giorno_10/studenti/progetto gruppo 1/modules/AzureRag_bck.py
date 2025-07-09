class AzureRAGSystem:
    """RAG system using Azure OpenAI for embeddings and generation"""
    
    def __init__(self, 
                 azure_endpoint: str,
                 api_key: str,
                 embedding_deployment: str = "text-embedding-ada-002",
                 gpt_deployment: str = "gpt-4",
                 api_version: str = "2024-02-01"):
        """
        Initialize the RAG system with Azure OpenAI
        
        Args:
            azure_endpoint: Your Azure OpenAI service endpoint
            api_key: API key for Azure OpenAI
            embedding_deployment: Deployment name for the embedding model
            gpt_deployment: Deployment name for GPT-4
            api_version: Azure OpenAI API version
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        self.embedding_deployment = embedding_deployment
        self.gpt_deployment = gpt_deployment
        
        # Initialize FAISS index for vector search
        self.index = None
        self.documents = []
        self.embeddings = []
        
        # Tokenizer for counting tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks with overlap to preserve context
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Find a natural break point (end of sentence)
            if end < len(text):
                # Look for the nearest separator
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
        Get the embedding for a text using Azure OpenAI
        
        Args:
            text: Text to convert to embedding
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedding_deployment,
            input=text
        )
        return response.data[0].embedding
    
    def add_documents(self, documents: List[Dict[str, str]], chunk_size: int = 1000):
        """
        Add documents to the RAG corpus
        
        Args:
            documents: List of documents with 'title' and 'content'
            chunk_size: Chunk size for long documents
        """
        all_chunks = []
        
        for doc in documents:
            # Chunk the document
            chunks = self.chunk_text(doc['content'], chunk_size)
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'title': doc['title'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'metadata': doc.get('metadata', {})
                })
        
        # Generate embeddings for all chunks
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = []
        
        for i, chunk in enumerate(all_chunks):
            print(f"Processed {i}/{len(all_chunks)} chunks")         
            embedding = self.get_embedding(chunk['text'])
            embeddings.append(embedding)
        
        # Update the FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        
        if self.index is None:
            # Create new index
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
        
        self.index.add(embeddings_array)
        self.documents.extend(all_chunks)
        self.embeddings.extend(embeddings)
        
        print(f"Added {len(all_chunks)} chunks to the index. Total documents: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for the most relevant documents for the query
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of tuples (document, score)
        """
        if self.index is None or len(self.documents) == 0:
            raise ValueError("No documents in the corpus. Use add_documents() first.")
        
        # Get embedding for the query
        query_embedding = np.array([self.get_embedding(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search for the most similar documents
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def generate_response(self, query: str, top_k: int = 3, max_tokens: int = 2000) -> Dict:
        """
        Generate a response using RAG
        
        Args:
            query: User question
            top_k: Number of documents to use as context
            max_tokens: Maximum number of tokens for the answer
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant documents
        relevant_docs = self.search(query, top_k)
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for doc, score in relevant_docs:
            context_parts.append(f"[Document: {doc['title']}]\n{doc['text']}")
            sources.append({
                'title': doc['title'],
                'score': score,
                'chunk_index': doc['chunk_index']
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Count tokens to ensure the limit is not exceeded
        context_tokens = len(self.encoding.encode(context))
        query_tokens = len(self.encoding.encode(query))
        
        # Leave space for the answer
        max_context_tokens = 6000  # Conservative limit for GPT-4
        if context_tokens + query_tokens > max_context_tokens:
            # Truncate context if needed
            context = self.encoding.decode(
                self.encoding.encode(context)[:max_context_tokens - query_tokens]
            )
        
        # Build the prompt
        system_prompt = """You are an expert assistant who answers questions based solely 
        on the provided context. If the requested information is not present in the context, 
        state this explicitly. Always cite which document the information comes from."""
        
        user_prompt = f"""Context:
        {context}

        Question: {query}

        Answer the question based only on the context above. 
        If you cite information, indicate which document it comes from."""
        
        # Generate the answer with GPT-4
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