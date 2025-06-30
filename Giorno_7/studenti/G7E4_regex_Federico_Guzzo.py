import re

def extract_ibans(text):
    """
    Estrae tutti gli IBAN presenti nel testo dato.

    Args:
        text (str): Testo in cui cercare gli IBAN.

    Returns:
        list: Lista di stringhe contenenti gli IBAN trovati.
    """
    iban_pattern = r"\bIT\d{2}[A-Z0-9]{1,23}\b"
    return re.findall(iban_pattern, text)

def main():
    text = (
        "Mario rossi ha ricevuto un bonifico sull'IBAN IT60X0542811101000000123456, "
        "dal suo IBAN IT60X054281110100A000678456."
    )
    
    ibans = extract_ibans(text)
    
    print("IBAN trovati:")
    for iban in ibans:
        print("IBAN:", iban)

if __name__ == "__main__":
    main()
