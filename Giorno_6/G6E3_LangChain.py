from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Imposta la tua API key OpenAI
os.environ["OPENAI_API_KEY"] = "sk-..."

# 1. Caricamento dei documenti
files = ["Mail.txt", "nota.txt", "fattura.txt"]
docs = []
for file in files:
    loader = TextLoader(file, encoding="utf-8")
    docs.extend(loader.load())

# 2. Suddivisione in chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Creazione degli embedding
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embedding, persist_directory="rag_db")

# 4. Retrieval + QA chain
retriever = vectordb.as_retriever()
llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5. Fai una domanda
query = "Qual è l'importo totale della fattura N. 123/2025?"
risposta = qa_chain(query)

# Output
print("Risposta:", risposta["result"])
print("\nFonte:", risposta["source_documents"][0].metadata['source'])

