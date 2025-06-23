import os
from dotenv import load_dotenv
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import List


load_dotenv()  

class TextFileLoader:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Percorso non trovato: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Non è una directory: {folder_path}")

    def load(self) -> list[dict]:
        """
        Legge tutti i file .txt nel percorso specificato.
        Ritorna: lista di dizionari con 'file_name' e 'content'
        """
        results = []
        for file in self.folder_path.glob("*.txt"):
            try:
                content = file.read_text(encoding="utf-8")
                results.append({
                    "file_name": file.name,
                    "content": content
                })
            except Exception as e:
                print(f"Errore nella lettura di {file.name}: {e}")
        return results
    

    
class AIProjectClientDefinition:
    def __init__(self):
        self.endpoint = os.getenv("PROJECT_ENDPOINT") 

        self.client = AIProjectClient(
            endpoint=self.endpoint,
            azure_endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
        )


class AdaEmbeddingModel(AIProjectClientDefinition):
    def __init__(self, client_class: AIProjectClientDefinition, model_name: str = "text-embedding-ada-002"):
        self.client = client_class.client
        self.model_name = model_name
        self.azure_client = client_class.client.inference.get_azure_openai_client(api_version="2023-05-15")


    def embed_text(self, text: str) -> list[float]:
        response = self.azure_client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding



class LangchainAdaWrapper(Embeddings):
    def __init__(self, ada_model: AdaEmbeddingModel):
        self.ada_model = ada_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.ada_model.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.ada_model.embed_text(text)
    



class ChatCompletionModel(AIProjectClientDefinition):
    def __init__(self, client_class: AIProjectClientDefinition, model_name: str = "gpt-4o", content: str = "", question: str = ""):
        self.client = client_class.client
        self.model_name = model_name
        self.azure_client = client_class.client.inference.get_azure_openai_client(api_version="2025-01-01-preview")
        self.content = content
        self.question = question


    def ask_about_document(self, content: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Sei un assistente AI specializzato nell'analisi e nella classificazione di documenti testuali."
                ),
            },
            {
                "role": "user",
                "content": f"Documento:\n{content}\n\nDomanda: {question}"
            }
        ]

        response = self.azure_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
        )
        return response.choices[0].message.content


class RAGPipeline:
    def __init__(self, folder_path: str):
        loader = TextFileLoader(folder_path)
        results = loader.load()

        ai_client = AIProjectClientDefinition()
        ada_model = AdaEmbeddingModel(ai_client)
        embedding_wrapper = LangchainAdaWrapper(ada_model)

        documents = [
            Document(page_content=doc["content"], metadata={"file_name": doc["file_name"]})
            for doc in results
        ]
        vectorstore = FAISS.from_documents(documents, embedding=embedding_wrapper)
        self.retriever = vectorstore.as_retriever()
        self.chat_model = ChatCompletionModel(ai_client)

    def answer_query(self, query: str) -> str:
        docs_simili = self.retriever.get_relevant_documents(query)
        if not docs_simili:
            return "Nessun documento rilevante trovato."

        risposte = ""
        for doc in docs_simili:
            risposta = self.chat_model.ask_about_document(doc.page_content, query)
            risposte += f"**{doc.metadata['file_name']}**\n{risposta}\n\n"
        return risposte


if __name__ == "__main__":
    folder_path = r"C:\DesktopNoOneDrive\ai-academy\Giorno_6\rag_documents"
    pipeline = RAGPipeline(folder_path)

    import gradio as gr
    with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
        gr.Markdown(
            """
            # 📄 RAG con Azure OpenAI
            Interroga i tuoi documenti locali con embeddings **Ada** e risposte da **GPT-4o**.  
            Inserisci una domanda qui sotto e ottieni risposte intelligenti basate sul contenuto.
            """,
            elem_id="title"
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="🧠 Domanda",
                    placeholder="Es. Qual è l’oggetto del documento X?",
                    lines=2,
                    interactive=True
                )
                submit_btn = gr.Button("Cerca tra i documenti", variant="primary")

            with gr.Column(scale=2):
                output_area = gr.Markdown(label="📚 Risposta")

        submit_btn.click(fn=pipeline.answer_query, inputs=query_input, outputs=output_area)

    demo.launch()