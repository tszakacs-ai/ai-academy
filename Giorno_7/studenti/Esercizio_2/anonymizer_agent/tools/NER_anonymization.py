from crewai.tools import tool
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline(
    "ner", 
    model=model, 
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)
    
@tool("ner_anonymization_tool")
def ner_anonymization_tool(text: str) -> str:
    """
    Anonymize personally identifiable information (PII) in text using Named Entity Recognition.
    
    Args:
        text: The input text to anonymize
        
    Returns:
        Anonymized text with PII replaced by appropriate placeholders
    """
    try:
        # Get NER results
        ner_results = ner_pipeline(text)
        
        # Define entity mappings for anonymization
        entity_replacements = {
            'PER': '[PERSON]',       # Person names
            'LOC': '[LOCATION]',     # Locations
            'ORG': '[ORGANIZATION]', # Organizations
            'MISC': '[MISC]'         # Miscellaneous entities
        }
        
        # Sort entities by start position in reverse order to avoid position shifts
        ner_results.sort(key=lambda x: x['start'], reverse=True)
        
        anonymized_text = text
        
        # Replace entities with anonymized placeholders
        for entity in ner_results:
            entity_type = entity['entity_group']
            start = entity['start']
            end = entity['end']
            confidence = entity['score']
            
            # Only anonymize if confidence is above threshold
            if confidence > 0.7:
                replacement = entity_replacements.get(entity_type, '[ENTITY]')
                anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
        
        return anonymized_text
        
    except Exception as e:
        return f"Error during anonymization: {str(e)}"