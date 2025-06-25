from ner_model import MultilingualNER, iban_detector
from pathlib import Path

if __name__ == "__main__":
    local_dir = Path(__file__).parent

    print("🤖 SCRIPT NER MULTILINGUALE")
    print("Modello: Davlan/bert-base-multilingual-cased-ner-hrl")
    print("="*60)

    with open(local_dir / "sample_data.csv", "r", encoding="utf-8") as f:
        target_text = f.readlines()[1:]  # Salta la prima riga
    
    ner = MultilingualNER()

    for i, text in enumerate(target_text, 1):
        print(f"\n--- ESEMPIO {i} ---")
        entities = ner.analyze_text(text, confidence_threshold=0.8)
        
        # Raggruppa per tipo
        entity_types = {}
        for entity in entities:
            entity_type = entity['entity_group']
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity['word'])
        
        if entity_types:
            print("📊 RIEPILOGO PER TIPO:")
            for entity_type, words in entity_types.items():
                print(f"   {entity_type}: {', '.join(set(words))}")
        
        for line in target_text:
            iban_results = iban_detector(line)
            print(f"\n🔍 RISULTATI IBAN: {iban_results}")