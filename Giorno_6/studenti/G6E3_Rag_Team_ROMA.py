import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

print("Avvio del sistema RAG con LangChain per interrogare i documenti.")

# Load environment variables from .env file
load_dotenv()

# --- Configuration (using environment variables for security) ---
AZURE_ENDPOINT = "https://testacademy3.openai.azure.com//"
AZURE_KEY = "1vnfa6KkzjJvWX2TJxBi0nxKotFYEZZp3X9zmIr6xQT5SVTveYZdJQQJ99BFACYeBjFXJ3w3AAABACOGGO9k"
AZURE_DEPLOYMENT = "gpt-4.1"
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
API_VERSION = "2024-02-15-preview"

DOCUMENTS_DIR = '.' # Usiamo la directory corrente per semplicità
CHROMA_DB_DIR = "./chroma_db" # Directory for Chroma DB persistence

def load_documents_and_create_vectorstore(directory: str):
    """
    Carica i documenti, li divide in chunk, genera embeddings e crea un vector store.
    """
    print(f"\nCaricamento e preparazione documenti dalla cartella: '{directory}'...")
    documents = []
    file_list = glob.glob(os.path.join(directory, "*.txt"))

    if not file_list:
        print(f"❌ Nessun file .txt trovato in '{directory}'. Assicurati che i file siano presenti.")
        return None

    for file_path in file_list:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Errore durante la lettura del file {os.path.basename(file_path)}: {e}")

    if not documents:
        print("Nessun documento valido è stato caricato.")
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"✅ {len(documents)} documenti caricati e suddivisi in {len(splits)} frammenti.")

    # Initialize Azure OpenAI Embeddings
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            openai_api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY
        )
        print("✅ Azure OpenAI Embeddings configurato.")
    except Exception as e:
        print(f"❌ Errore nella configurazione di Azure OpenAI Embeddings: {e}")
        return None

    # Create a vector store (Chroma in this case)
    print("Creazione o caricamento del Vector Store Chroma...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist() # Save the vector store to disk
    print(f"✅ Vector Store Chroma creato/caricato e persistito in '{CHROMA_DB_DIR}'.")
    return vectorstore

def start_rag_chat():
    """
    Avvia la chat interattiva per interrogare i documenti usando LangChain.
    """
    vectorstore = load_documents_and_create_vectorstore(DOCUMENTS_DIR)
    if not vectorstore:
        return

    # Initialize Azure OpenAI Chat Model
    try:
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            temperature=0.0 # Consistent with your original temperature
        )
        print("✅ Azure OpenAI Chat Model configurato e pronto.")
    except Exception as e:
        print(f"❌ Errore nella configurazione del client AzureChatOpenAI: {e}")
        return

    # Define the prompt for combining documents
    # LangChain automatically injects 'context' and 'input' based on the chain type
    prompt = ChatPromptTemplate.from_template("""
    Sei un assistente AI specializzato nell'analisi di documenti.
    Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sulle informazioni contenute nel seguente contesto.
    Il contesto è una raccolta di più documenti.
    Se l'informazione non è presente nel contesto, rispondi con "L'informazione non è presente nei documenti forniti."
    Non inventare nulla. Sii preciso e cita il nome del documento da cui prendi l'informazione, se possibile.

    Contesto:
    {context}

    Domanda: {input}
    """)

    # Create a document combining chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Create the RAG chain
    # This chain will:
    # 1. Retrieve relevant documents based on the user's question.
    # 2. Pass those documents and the question to the document_chain to generate the answer.
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\n\n--- Chat con i tuoi Documenti (LangChain) ---")
    print("Puoi iniziare a fare domande. Scrivi 'exit' o 'quit' per uscire.")

    while True:
        user_question = input("\nLa tua domanda: ")

        if user_question.lower() in ['exit', 'quit', 'esci']:
            print("Arrivederci!")
            break

        try:
            print("\nSto pensando...")
            # Invoke the RAG chain
            response = retrieval_chain.invoke({"input": user_question})

            # The answer is in the "answer" key of the response
            answer = response["answer"]
            print("\nRisposta:")
            print(answer)

            # You can also inspect the retrieved documents if needed:
            # for doc in response["context"]:
            #     print(f"\n--- Retrieved from: {doc.metadata.get('source', 'Unknown')} ---")
            #     print(doc.page_content[:200] + "...")

        except Exception as e:
            print(f"\n❌ Si è verificato un errore durante la chiamata alla catena RAG: {e}")
            # print(f"Dettagli errore: {e.args}") # For more detailed error info

if __name__ == "__main__":
    # Ensure your environment variables are set before running:
    # AZURE_OPENAI_ENDPOINT
    # AZURE_OPENAI_API_KEY
    # AZURE_OPENAI_CHAT_DEPLOYMENT_NAME (for your LLM)
    # AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME (for your embeddings)
    # AZURE_OPENAI_API_VERSION (e.g., 2024-02-01)
    start_rag_chat()
