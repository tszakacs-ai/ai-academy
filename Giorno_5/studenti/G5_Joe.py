import os
import json
import re
import ssl
from typing import Dict, List, Any
from transformers import pipeline
from dotenv import load_dotenv
import openai

# SSL workaround (only if needed for custom certs)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["REQUESTS_CA_BUNDLE"] = r"C:\\Users\\SE645QY\\custom-ca-bundle.pem"

def unique_case_insensitive(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            result.append(item)
    return result

class NERExtractor:
    def __init__(self):
        self.ner = pipeline(
            task="token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device_map="auto"
        )

    def extract(self, text: str) -> Dict[str, List[str]]:
        categories = {
            "PERSON": [],
            "ORG": [],
            "LOC": [],
            "DATE": [],
            "BANKCODE": [],
            "TAXCODE": [],
            "MISC": []
        }
        for entity in self.ner(text):
            label = entity["entity_group"]
            word = entity["word"].strip()
            if label == "PER":
                categories["PERSON"].append(word)
            elif label == "ORG":
                categories["ORG"].append(word)
            elif label == "LOC":
                categories["LOC"].append(word)
            elif label == "DATE":
                categories["DATE"].append(word)
            elif label == "BANKCODE":
                categories["BANKCODE"].append(word)
            elif label == "TAXCODE":
                categories["TAXCODE"].append(word)
            else:
                categories["MISC"].append(word)
        # Remove duplicates (case-insensitive)
        for key in categories:
            categories[key] = unique_case_insensitive(categories[key])
        return categories

class EmailAnonymizer:
    def __init__(self, input_dir: str = "input_documents", output_dir: str = "output_documents"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        load_dotenv()
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        self.openai_client = openai.AzureOpenAI(
            api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            api_version="2024-12-01-preview",
        )
        self.ner_extractor = NERExtractor()

    @staticmethod
    def log(label: str, content: Any):
        print(f"\n[LOG] {label}\n{'-' * 60}")
        if isinstance(content, (dict, list)):
            print(json.dumps(content, indent=2, ensure_ascii=False))
        else:
            print(str(content))

    @staticmethod
    def replace_entities(text: str, entities: Dict[str, List[str]]) -> str:
        result = text
        for label, values in entities.items():
            for value in values:
                if value:
                    result = re.sub(re.escape(value), f"[{label}]", result)
        return result

    def anonymize_file(self, filename: str):
        path = os.path.join(self.input_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.log("Original Text (truncated)", text[:500] + ("..." if len(text) > 500 else ""))

        entities = self.ner_extractor.extract(text)
        self.log("Extracted Entities", entities)

        if not any(entities.values()):
            print(f"[INFO] No entities found in {filename}. Skipping.")
            return

        local_anonymized = self.replace_entities(text, entities)
        self.log("Locally Anonymized (truncated)", local_anonymized[:500] + ("..." if len(local_anonymized) > 500 else ""))

        prompt = (
            "You are a privacy assistant helping to redact sensitive information from documents.\n"
            "Below is a document that has already been partially anonymized.\n"
            "Use the list of named entities provided to ensure all sensitive data has been fully replaced with appropriate placeholders (e.g., [PERSON], [IBAN], [ORG], etc.).\n"
            "Do not attempt to reconstruct or guess any original information.\n"
            "--- Entities extracted ---\n"
            f"{entities}\n"
            "--- Anonymized Document ---\n"
            f"{local_anonymized}\n"
            "Please return only the fully anonymized version of the document."
        )
        self.log("Prompt to GPT-4o (truncated)", prompt[:500] + "...")

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data-privacy assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=2024,
                temperature=0.3
            )
            final_text = response.choices[0].message.content
            self.log("GPT-4o Response (truncated)", final_text[:500] + ("..." if len(final_text) > 500 else ""))

            output_name = os.path.splitext(filename)[0] + "_anonymized.txt"
            output_path = os.path.join(self.output_dir, output_name)
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(final_text)
            print(f"[SUCCESS] File saved to: {output_path}")
        except Exception as exc:
            print(f"[ERROR] Failed to process {filename}: {exc}")

    def run(self):
        for fname in os.listdir(self.input_dir):
            if fname.endswith(".txt"):
                print(f"\n[INFO] Processing file: {fname}")
                self.anonymize_file(fname)

if __name__ == "__main__":
    EmailAnonymizer().run()
