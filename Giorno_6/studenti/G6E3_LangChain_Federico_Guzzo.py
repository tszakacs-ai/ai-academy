import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


def load_documents(file_paths):
    print("📄 Caricamento documenti...")
    documents = []
    for file in file_paths:
        if os.path.exists(file):
            loader = TextLoader(file, encoding="utf-8")
            documents.extend(loader.load())
            print(f"✅ Caricato: {file}")
        else:
            print(f"⚠️ File non trovato: {file}")
    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    print("🔪 Suddivisione in chunk...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"🧩 Totale chunk creati: {len(chunks)}")
    return chunks


def create_vectorstore(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"🔎 Creazione embeddings con modello: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb


def create_llm(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_tokens=256):
    print(f"🤖 Caricamento LLM: {model_id}")
    generation_pipeline = pipeline("text-generation", model=model_id, max_new_tokens=max_tokens)
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    return llm


def ask_question(qa_chain, query):
    print(f"\n❓ Domanda: {query}")
    response = qa_chain.invoke({"query": query})
    print("\n🧠 Risposta:", response["result"])
    print("\n📚 Fonti:")
    for doc in response["source_documents"]:
        print("-", doc.metadata["source"])


if __name__ == "__main__":
    print("🛠️  Avvio script NLP - Federico Guzzo\n")

    # 1. File di input
    files = ["studenti/RAGDocs/Mail.txt", "studenti/RAGDocs/nota.txt", "studenti/RAGDocs/Fattura.txt"]

    # 2. Carica documenti
    docs = load_documents(files)

    # 3. Chunking
    chunks = split_documents(docs)

    # 4. Embedding e indicizzazione
    vectordb = create_vectorstore(chunks)
    retriever = vectordb.as_retriever()

    # 5. LLM Hugging Face
    llm = create_llm()

    # 6. RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # 7. Domanda personalizzata
    domanda = "Qual è il nuovo IBAN comunicato da Mario Rossi?"
    ask_question(qa_chain, domanda)
