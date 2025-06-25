import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Chat completion model credentials
azure_chat_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_chat_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_chat_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_chat_deployment = os.getenv("AZURE_CHAT_DEPLOYMENT")

# Embedding model credentials
azure_embedding_api_key = os.getenv("AZURE_EMBEDDING_API_KEY")  
azure_embedding_endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT") 
azure_embedding_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION")
azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

# Validate credentials
if azure_chat_api_key and azure_chat_endpoint:
    print("Azure Chat API credentials loaded successfully")
else:
    print("Failed to load Azure Chat API credentials - please check your .env file")

if azure_embedding_api_key and azure_embedding_endpoint:
    print("Azure Embedding API credentials loaded successfully")
else:
    print("Failed to load Azure Embedding API credentials - please check your .env file")

def create_rag_pipeline():
    """
    Create a RAG pipeline using LangChain with the following steps:
    1. Document loading
    2. Chunking
    3. Embedding generation
    4. Retrieval
    5. LLM integration for responses
    """
    
    # Step 1: Document loading
    print("Step 1: Loading documents...")
    documents = []
    
    # Load PDF files from a 'documents' directory
    data_dir = os.path.join(os.path.dirname(__file__), "documents")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory {data_dir}. Please add your documents there.")
        return
        
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            text_path = os.path.join(data_dir, file)
            loader = TextLoader(text_path)
            documents.extend(loader.load())
    
    if not documents:
        print("No documents found. Please add PDF or TXT files to the documents directory.")
        return
    
    
    print(f"Loaded {len(documents)} document(s).")
    
    # Step 2: Splitting into chunks
    print("Step 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    # Step 3: Creating embeddings using Azure
    print("Step 3: Creating embeddings using Azure OpenAI...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_embedding_deployment,
        openai_api_key=azure_embedding_api_key,
        azure_endpoint=azure_embedding_endpoint,
        api_version=azure_embedding_api_version
    )
    
    # Create vector store
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("Vector store created successfully")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        # Maybe try without persistence to see if that's the issue
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
    
    # Step 4: Set up retrieval mechanism
    print("Step 4: Setting up retrieval mechanism...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Step 5: Set up the Azure language model
    print("Step 5: Configuring Azure OpenAI language model...")
    llm = AzureChatOpenAI(
        deployment_name=azure_chat_deployment,
        openai_api_key=azure_chat_api_key,
        azure_endpoint=azure_chat_endpoint,
        api_version=azure_chat_api_version,
        temperature=1
    )
    
    # Create RAG chain
    print("Setting up RAG pipeline...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def query_rag_system(qa_chain, query):
    """
    Query the RAG system with a user question
    """
    if not qa_chain:
        print("RAG pipeline not initialized.")
        return
    
    print(f"\nQuerying: '{query}'")
    result = qa_chain.invoke({"query": query})
    
    print("\nAnswer:")
    print(result["result"])
    
    print("\nSource documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Document {i+1}:")
        print(f"Source: {doc.metadata}")
        print(f"Content: {doc.page_content[:150]}...")
        print("-" * 50)

def main():
    print("Initializing RAG Pipeline with LangChain using Azure OpenAI")
    print("=" * 50)
    
    # Create the RAG pipeline
    qa_chain = create_rag_pipeline()
    
    if qa_chain:
        print("\nRAG pipeline created successfully!\n")
        
        # Interactive query loop
        while True:
            query = input("\nEnter a question (or 'exit' to quit): ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            query_rag_system(qa_chain, query)

if __name__ == "__main__":
    main()