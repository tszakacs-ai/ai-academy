import pandas as pd

def normalize_entities(input_path, output_path):
    df = pd.read_csv(input_path)
    # Esempio: normalizza nomi aziende e date
    if 'Azienda' in df.columns:
        df['Azienda'] = df['Azienda'].str.upper().str.replace(' SRL', ' Srl')
    if 'Data' in df.columns:
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce').dt.strftime('%Y-%m-%d')
    df.to_csv(output_path, index=False)
    print(f"File normalizzato salvato in {output_path}")

if __name__ == "__main__":
    normalize_entities("../data/entities_clean.csv", "../data/entities_normalized.csv")