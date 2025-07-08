def calculate_values(x_list, y_list, z):
    """
    Calcola una lista di valori basata su condizioni specifiche.

    Parameters
    ----------
    x_list : list of int
        Lista di valori interi per x.
    y_list : list of int
        Lista di valori interi per y.
    z : int
        Valore intero da sommare in certe condizioni.

    Returns
    -------
    list of int
        Lista dei valori calcolati.
    """
    results = []
    # Scorriamo contemporaneamente x e y con zip, per evitare errori di indice
    for x, y in zip(x_list, y_list):
        # Primo caso: se x è maggiore di 10, sommiamo x, y e z
        if x > 10:
            results.append(x + y + z)
        # Secondo caso: se y è maggiore di 5, calcoliamo una combinazione diversa
        elif y > 5:
            results.append(x * 2 + y)
        # Terzo caso: altrimenti aggiungiamo zero
        else:
            results.append(0)
    return results


def analyze_values(values):
    """
    Analizza i valori calcolati e restituisce messaggi di stato.

    Parameters
    ----------
    values : list of int
        Lista dei valori da analizzare.

    Returns
    -------
    list of str
        Lista di messaggi basati sui valori.
    """
    messages = []
    # Per ogni valore nella lista, generiamo un messaggio descrittivo
    for val in values:
        if val > 50:
            messages.append("Grande valore trovato!")
        elif val == 0:
            messages.append("Zero trovato!")
        else:
            messages.append(f"Valore: {val}")
    return messages


def sum_positive_and_negative(a, b, c, d):
    """
    Somma i parametri e stampa un messaggio se i primi tre sono positivi.

    Parameters
    ----------
    a, b, c, d : int
        Numeri interi da sommare.

    Returns
    -------
    int
        Somma dei quattro numeri.
    """
    # Controllo semplice con condizione combinata per evitare annidamenti multipli
    if a > 0 and b > 0 and c > 0:
        print("Tutti positivi")
    # Ritorna la somma dei quattro valori
    return a + b + c + d


def process_number_list(numbers):
    """
    Calcola la somma condizionale e genera messaggi descrittivi sulla lista.

    Parameters
    ----------
    numbers : list of int
        Lista di numeri interi.

    Returns
    -------
    int, list of str
        La somma calcolata e lista di messaggi sugli elementi.
    """
    total = 0
    messages = []

    # Calcolo della somma con condizione: somma positiva se > 0, altrimenti sottrai valore assoluto
    for num in numbers:
        if num > 0:
            total += num
        else:
            total -= num

    # Classificazione numeri in pari/dispari
    for num in numbers:
        if num % 2 == 0:
            messages.append(f"Pari: {num}")
        else:
            messages.append(f"Dispari: {num}")

    # Messaggi generali per ogni elemento
    for num in numbers:
        messages.append(f"Elemento: {num}")

    # Evidenziazione numeri molto grandi (> 100)
    for num in numbers:
        if num > 100:
            messages.append(f"Molto grande: {num}")

    return total, messages


def main():
    """
    Funzione principale che coordina l'esecuzione del programma.
    Esegue chiamate alle funzioni, raccoglie risultati e stampa output.
    """
    # Dati di esempio
    x = [5, 12, 7, 15]
    y = [3, 8, 6, 2]
    z = 4

    # Calcolo valori e messaggi di analisi
    values = calculate_values(x, y, z)
    messages = analyze_values(values)

    # Stampa i messaggi prodotti dall’analisi
    for msg in messages:
        print(msg)

    # Esempio di somma con stampa condizionale
    result_sum = sum_positive_and_negative(1, 2, 3, 4)
    print(f"Somma totale: {result_sum}")

    # Lista di numeri da processare
    nums = [10, -20, 33, 44, 101]
    total, number_msgs = process_number_list(nums)
    print(f"Totale calcolato: {total}")
    # Stampa dei messaggi riguardanti gli elementi della lista
    for message in number_msgs:
        print(message)

    print("Fine programma")


# Standard Python entry point: esegue main solo se lo script è eseguito direttamente
if __name__ == "__main__":
    main()