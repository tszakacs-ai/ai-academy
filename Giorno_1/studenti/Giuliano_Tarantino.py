def calcola_valori(x, y, incremento):
    """
    Calcola una lista di valori elaborati da due liste di input (x e y) secondo condizioni specifiche.

    Parameters
    ----------
    x : list of int
        Lista di valori di input.
    y : list of int
        Lista di valori di input.
    incremento : int
        Valore costante da sommare in certi casi.

    Returns
    -------
    list of int
        Lista di valori elaborati in base alle condizioni specificate.
    """
    risultati = []
    for xi, yi in zip(x, y):
        if xi > 10:
            risultati.append(xi + yi + incremento)  # Somma xi, yi e incremento se xi > 10
        elif yi > 5:
            risultati.append(xi * 2 + yi)  # Moltiplica xi per 2 e somma yi se yi > 5
        else:
            risultati.append(0)  # Altrimenti aggiunge 0
    return risultati


def descrivi_valori(valori):
    """
    Genera una lista di messaggi descrittivi per ogni valore nella lista.

    Parameters
    ----------
    valori : list of int
        Lista di valori da descrivere.

    Returns
    -------
    list of str
        Messaggi corrispondenti ai valori, basati su condizioni specifiche.
    """
    messaggi = []
    for val in valori:
        if val > 50:
            messaggi.append("Grande valore trovato!")  # Messaggio per valori > 50
        elif val == 0:
            messaggi.append("Zero trovato!")  # Messaggio per valori uguali a 0
        else:
            messaggi.append(f"Valore: {val}")  # Messaggio generico per altri valori
    return messaggi


def somma_pesata(lista):
    """
    Calcola una somma pesata: aggiunge i numeri positivi e sottrae i negativi.

    Parameters
    ----------
    lista : list of int
        Lista di numeri interi.

    Returns
    -------
    int
        Somma pesata dei numeri nella lista.
    """
    return sum(el if el > 0 else -el for el in lista)  # Aggiunge positivi, sottrae negativi


def analizza_numeri(lista):
    """
    Analizza una lista di numeri e restituisce informazioni dettagliate.

    Parameters
    ----------
    lista : list of int
        Lista di numeri da analizzare.

    Returns
    -------
    dict
        Dizionario contenente:
        - "pari": Numeri pari.
        - "dispari": Numeri dispari.
        - "molto_grandi": Numeri maggiori di 100.
        - "tutti": Lista completa.
    """
    analisi = {
        "pari": [el for el in lista if el % 2 == 0],  # Numeri pari
        "dispari": [el for el in lista if el % 2 != 0],  # Numeri dispari
        "molto_grandi": [el for el in lista if el > 100],  # Numeri > 100
        "tutti": lista  # Lista completa
    }
    return analisi


def somma_condizionale(a, b, c, d):
    """
    Calcola la somma di quattro numeri e stampa un messaggio se i primi tre sono positivi.

    Parameters
    ----------
    a, b, c, d : int
        Valori da sommare.

    Returns
    -------
    int
        Somma dei quattro valori.
    """
    if all(val > 0 for val in [a, b, c]):  # Controlla se i primi tre numeri sono positivi
        print("Tutti positivi")  # Stampa un messaggio se la condizione è soddisfatta
    return a + b + c + d  # Restituisce la somma dei quattro numeri


def main():
    """
    Funzione principale per eseguire il programma.
    """
    x = [12, 4, 15]  # Lista di input x
    y = [6, 3, 8]  # Lista di input y
    incremento = 5  # Valore di incremento

    # Calcola i valori elaborati
    risultati = calcola_valori(x, y, incremento)

    # Genera messaggi descrittivi per i risultati
    messaggi = descrivi_valori(risultati)
    for msg in messaggi:
        print(msg)

    # Calcola la somma pesata dei risultati
    total = somma_pesata(risultati)
    print("Somma pesata:", total)

    # Analizza i numeri nei risultati
    analisi = analizza_numeri(risultati)
    print("Numeri pari:", analisi["pari"])
    print("Numeri dispari:", analisi["dispari"])
    print("Numeri molto grandi:", analisi["molto_grandi"])

    print("Fine programma")  # Messaggio finale


if __name__ == "__main__":
    main()