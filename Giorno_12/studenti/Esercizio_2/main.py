import pandas as pd
from pathlib import Path
from svm_classifier import *


if __name__ == "__main__":
    parent_dir = Path(__file__).parent
    input_file = parent_dir / "data" / "spam.csv"

    df = pd.read_csv(input_file, delimiter=',', encoding='latin-1')

    # Fase esplorativa
    # print(df.info)
    # print(df.columns)
    # print(df.isnull().sum())

    # colonne_interesse = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    # sub_df = df[colonne_interesse]

    # for idx, row in sub_df.iterrows():
    #     for col in colonne_interesse:
    #         if pd.notna(row[col]):
    #             print(f"  {col}: {row[col]}")

    cleaned_df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    cleaned_df['label'] = cleaned_df['label'].map({'ham': 0, 'spam': 1})
     
    # Addestra il modello
    classifier, X_test, y_test = train_spam_classifier(cleaned_df)
    
    # Caricamento dei nuovi messaggi da classificare
    with open(parent_dir / "data" / "test.txt", "r", encoding="utf-8") as f:
        new_messages = f.readlines()
    new_messages = [msg.strip() for msg in new_messages if msg.strip()]
    
    results = classify_new_messages(classifier, new_messages)
    
    print("\nClassificazione di nuovi messaggi:")
    print("="*50)
    for message, prediction, confidence in results:
        print(f"Messaggio: {message[:50]}...")
        print(f"Classificazione: {prediction} (Confidenza: {confidence:.1f}%)")
        print("-" * 30)