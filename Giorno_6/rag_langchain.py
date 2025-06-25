import os
from dotenv import load_dotenv

load_dotenv()

# ---- LangChain + Azure Setup ----
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

# Azure client for embeddings
endpoint = os.getenv("ADA_ENDPOINT")
project = AIProjectClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential()
)

client = project.inference.get_azure_openai_client(api_version="2023-05-15")

# ---- Custom Embedding Model ----
from langchain.embeddings.base import Embeddings

class FoundryEmbedding(Embeddings):
    def __init__(self, client, model_name="text-embedding-ada-002"):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )
        return response.data[0].embedding

embedding_model = FoundryEmbedding(client)

# ---- Caricamento dei documenti da cartella ----
from langchain_core.documents import Document
folder_path = "C:/desktopnodrive/ai-academy/Giorno_6/documenti"
docs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(Document(page_content=text, metadata={"source": filename}))

print(f"Caricati {len(docs)} documenti.")

# ---- FAISS Vector Store corretto ----
import faiss
import numpy as np
import uuid
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore


# Step 1: embedding dei contenuti
texts = [doc.page_content for doc in docs]
embeddings = embedding_model.embed_documents(texts)

# Step 2: FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Step 3: associazione documenti
index_to_docstore_id = {i: str(uuid.uuid4()) for i in range(len(docs))}
docstore = InMemoryDocstore({doc_id: doc for doc_id, doc in zip(index_to_docstore_id.values(), docs)})

# Step 4: creazione FAISS store
faiss_store = FAISS(
    embedding_function=embedding_model.embed_query,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)
retriever = faiss_store.as_retriever()

# ---- RAG Chain ----
from langchain.chains import RetrievalQA
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.markdown import Markdown
from rich import box

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# ---- Loop Interattivo ----
console = Console()
console.rule("[bold blue]RAG Assistant Attivo[/bold blue]", style="bold blue")
console.print(
    Panel(
        Text("Digita una domanda sui tuoi documenti.\nScrivi 'quit' per uscire.", style="bold white"),
        box=box.DOUBLE,
        style="bold magenta",
        expand=False,
    )
)

while True:
    query = Prompt.ask("[bold cyan]Tu[/bold cyan]", console=console)
    if query.lower() in ["quit", "esci", "exit"]:
        console.print(Panel("[bold red]👋 Fine sessione.[/bold red]", expand=False, box=box.ROUNDED))
        break
    response = qa_chain.invoke(query)
    answer = response["result"] if isinstance(response, dict) and "result" in response else str(response)
    console.print(
        Panel(
            Markdown(answer),
            title="[bold yellow]Assistant[/bold yellow]",
            style="bold green",
            box=box.HEAVY,
            expand=False,
        )
    )

