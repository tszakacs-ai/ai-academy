from pathlib import Path
from typing import Iterable, List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:
    """Load PDF and text files into LangChain Documents."""

    def __init__(self, tmp_folder: str | Path) -> None:
        self.tmp_folder = Path(tmp_folder)
        self.tmp_folder.mkdir(exist_ok=True)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def load_files(self, uploaded_files: Iterable) -> List[Document]:
        documents: List[Document] = []
        for file in uploaded_files:
            if file.name.endswith(".txt"):
                content = file.getvalue().decode("utf-8")
                documents.append(Document(page_content=content, metadata={"file_name": file.name}))
            elif file.name.endswith(".pdf"):
                tmp_path = self.tmp_folder / file.name
                with open(tmp_path, "wb") as f:
                    f.write(file.getvalue())
                loader = PyPDFLoader(str(tmp_path))
                pages = loader.load()
                docs = self.splitter.split_documents(pages)
                for doc in docs:
                    documents.append(Document(page_content=doc.page_content, metadata={"file_name": file.name}))
        return documents
