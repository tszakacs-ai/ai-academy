from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# 1. Caricamento dei documenti
# Carica i file di testo (es. email, note, fatture) e li combina in una lista di documenti
files = ["Giorno_6/Mail.txt", "Giorno_6/nota.txt", "Giorno_6/Fattura.txt"]
documents = []
for file in files:
    loader = TextLoader(file, encoding="utf-8")
    documents.extend(loader.load())

# 2. Suddivisione dei documenti in chunk
# Divide i documenti in blocchi più piccoli per migliorare la gestione e la ricerca
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Creazione degli embedding
# Converte i chunk in vettori numerici utilizzando un modello di embedding pre-addestrato
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(chunks, embedding_model)

# 4. Configurazione del retriever
# Configura un sistema di recupero basato sugli embedding per cercare informazioni nei documenti
retriever = vectordb.as_retriever()

# 5. Inizializzazione del modello LLM
# Utilizza un modello di generazione testo locale (es. TinyLlama) per rispondere alle domande
llm_pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 6. Configurazione della catena Retrieval + QA
# Combina il retriever e il modello LLM per creare una catena di domande e risposte
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Esecuzione della query
# Formula una domanda specifica per ottenere informazioni dai documenti caricati
query = "Qual è il nuovo IBAN comunicato da Mario Rossi?"
risposta = qa_chain.invoke({"query": query})

# 8. Output dei risultati
# Stampa la risposta generata e le fonti dei documenti utilizzati
print("Risultato:", risposta["result"])
print("\nFonte:")
for doc in risposta["source_documents"]:
    print("-", doc.metadata["source"])
