import ssl
import os
import json
from transformers import pipeline

# SSL
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["REQUESTS_CA_BUNDLE"] = r"C:\\Users\\SE645QY\\custom-ca-bundle.pem"

class NERExtractor:
    def __init__(self):
        """Initialize the NER pipeline using a lightweight English model."""

        # English NER model dslim/bert-base-NER
        self.ner_pipe = pipeline(
            task="token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device_map="auto"
        )

    def extract_entities(self, text: str) -> dict:
        """
        Extract named entities from a given text using the NER pipeline.

        Parameters
        ----------
            text (str): 
            The input text to analyze.

        Returns
        -------
            dict: 
            A dictionary with keys representing entity types (e.g., PERSON, ORG, LOC, DATE, BANKCODE, TAXCODE, MISC),
                  and values as lists of unique detected entities.
        """
        result = {
            "PERSON": [],
            "ORG": [],
            "LOC": [],
            "DATE": [],
            "BANKCODE": [],
            "TAXCODE": [],
            "MISC": []
        }

        for ent in self.ner_pipe(text):
            tag = ent["entity_group"]
            word = ent["word"].strip()

            if tag == "PER":
                result["PERSON"].append(word)
            elif tag == "ORG":
                result["ORG"].append(word)
            elif tag == "LOC":
                result["LOC"].append(word)
            elif tag == "DATE":
                result["DATE"].append(word)
            elif tag == ["BANKCODE"]:
                result["BANKCODE"].append(word)
            elif tag == ["TAXCODE"]:
                result["TAXCODE"].append(word)
            else: 
                tag == "MISC"
                result["MISC"].append(word)

        for key in result:
            seen = set()
            result[key] = [x for x in result[key] if not (x.lower() in seen or seen.add(x.lower()))]

        return result

    def extract_from_file(self, file_path: str) -> dict:
        """
        Load text from a file and extract entities using the NER pipeline.

        Parameters
        ----------
            file_path (str): 
            Path to the text file containing input data.

        Returns
        -------
            dict: 
            Dictionary of extracted named entities grouped by type.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.extract_entities(text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract NER entities from a text file and output JSON.")
    parser.add_argument("document_path", nargs="?", default="document.txt", help="Path to the input document.txt file")
    args = parser.parse_args()
    extractor = NERExtractor()
    entities = extractor.extract_from_file(args.email_path)
    print(json.dumps(entities, ensure_ascii=False, indent=2))
