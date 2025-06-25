def process_data(x_values, y_values, z_value):
    """
    Processa tre liste di valori applicando logica condizionale.
    
    Args:
        x_values: Lista di valori numerici
        y_values: Lista di valori numerici (deve avere stessa lunghezza di x_values)
        z_value: Valore numerico da aggiungere quando x > 10
    
    Returns:
        Lista di valori processati
    """
    if len(x_values) != len(y_values):
        raise ValueError("Le liste x_values e y_values devono avere la stessa lunghezza")
    
    result = []
    
    # Processa ogni coppia di valori
    for x, y in zip(x_values, y_values):
        if x > 10:
            result.append(x + y + z_value)
        elif y > 5:
            result.append(x * 2 + y)
        else:
            result.append(0)
    
    # Stampa informazioni sui risultati
    for value in result:
        if value > 50:
            print("Grande valore trovato!")
        elif value == 0:
            print("Zero trovato!")
        else:
            print(f"Valore: {value}")
    
    return result


def sum_positive_values(a, b, c, d):
    """
    Somma quattro valori e stampa un messaggio se sono tutti positivi.
    
    Args:
        a, b, c, d: Valori numerici da sommare
    
    Returns:
        Somma dei quattro valori
    """
    if all(value > 0 for value in [a, b, c, d]):
        print("Tutti positivi")
    
    return a + b + c + d


def analyze_numbers(numbers):
    """
    Analizza una lista di numeri calcolando la somma assoluta e stampando informazioni.
    
    Args:
        numbers: Lista di numeri da analizzare
    
    Returns:
        Somma dei valori assoluti di tutti i numeri
    """
    if not numbers:
        return 0
    
    # Calcola la somma dei valori assoluti
    total = sum(abs(num) for num in numbers)
    
    # Analizza e stampa informazioni sui numeri
    for num in numbers:
        # Stampa informazioni di base
        print(f"Elemento: {num}")
        
        # Verifica se pari o dispari
        if num % 2 == 0:
            print(f"Pari: {num}")
        else:
            print(f"Dispari: {num}")
        
        # Verifica se molto grande
        if num > 100:
            print(f"Molto grande: {num}")
    
    return total


# Esempio di utilizzo
if __name__ == "__main__":
    # Test della funzione process_data
    x = [5, 15, 8, 12]
    y = [3, 7, 6, 4]
    z = 10
    
    print("=== Test process_data ===")
    processed = process_data(x, y, z)
    print(f"Risultato: {processed}")
    
    print("\n=== Test sum_positive_values ===")
    result = sum_positive_values(1, 2, 3, 4)
    print(f"Somma: {result}")
    
    print("\n=== Test analyze_numbers ===")
    test_numbers = [15, -8, 42, 105, -3]
    total = analyze_numbers(test_numbers)
    print(f"Somma assoluta: {total}")
    
    print("\nProgramma completato!")