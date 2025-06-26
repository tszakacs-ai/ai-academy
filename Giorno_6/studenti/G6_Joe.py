import os
from dotenv import load_dotenv
from openai import AzureOpenAI

class DocumentClassifier:
    ALLOWED_TYPES = ["credit note", "purchase order", "contract", "email", "unknown"]

    def __init__(self, documents_dir: str, model: str = "gpt-4o"):
        load_dotenv()
        self.documents_dir = documents_dir
        self.model = model
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY_4o"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT_4o"),
            api_version="2024-12-01-preview",
        )

    @staticmethod
    def build_prompt(doc_text: str) -> str:
        allowed = ", ".join(DocumentClassifier.ALLOWED_TYPES)
        return (
            "You will receive the full text of a company document.\n"
            f"You have to return only the type of document, exactly as one of the options:\n{allowed}.\n"
            f'"""{doc_text}"""'
        )

    def classify_document(self, doc_text: str) -> str:
        prompt = self.build_prompt(doc_text)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a document classification assistant specialised in classifying company documents."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def run(self):
        for filename in os.listdir(self.documents_dir):
            path = os.path.join(self.documents_dir, filename)
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as fp:
                doc_text = fp.read()
            doc_type = self.classify_document(doc_text)
            print(f"Input file: {filename} -> Classified type: {doc_type}")

if __name__ == "__main__":
    classifier = DocumentClassifier(documents_dir=r"Giorno_5\RAGDocs")
    classifier.run()
