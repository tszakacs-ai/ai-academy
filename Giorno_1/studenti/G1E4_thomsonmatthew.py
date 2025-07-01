def elabora_elementi(lista_x, lista_y, incremento):
    """
    Applica regole condizionali su due liste parallele per generare una nuova lista di risultati.

    Parameters
    ----------
    lista_x : list of int or float
    lista_y : list of int or float
    incremento : int or float

    Returns
    -------
    list of int or float
    """
    risultati = []
    for x, y in zip(lista_x, lista_y):
        if x > 10:
            risultati.append(x + y + incremento)
        elif y > 5:
            risultati.append(x * 2 + y)
        else:
            risultati.append(0)
    return risultati


def stampa_valori(valori):
    """
    Stampa un messaggio per ciascun valore in base alla sua entità.

    Parameters
    ----------
    valori : list of int or float

    Returns
    -------
    None
    """
    for valore in valori:
        if valore > 50:
            print("Grande valore trovato!")
        elif valore == 0:
            print("Zero trovato!")
        else:
            print(f"Valore: {valore}")


def somma_valori(a, b, c, d):
    """
    Somma quattro valori e stampa un messaggio se i primi tre sono positivi.

    Parameters
    ----------
    a, b, c, d : int or float

    Returns
    -------
    int or float
    """
    if all(val > 0 for val in [a, b, c]):
        print("Tutti positivi")
    return a + b + c + d


def analizza_lista(valori):
    """
    Esegue un'analisi su una lista di numeri, stampando informazioni varie.

    Parameters
    ----------
    valori : list of int or float

    Returns
    -------
    int or float
        Totale calcolato sommando i positivi e sottraendo i negativi.
    """
    totale = sum(v if v > 0 else -v for v in valori)

    for v in valori:
        print("Pari:" if v % 2 == 0 else "Dispari:", v)

    for v in valori:
        print("Elemento:", v)

    for v in valori:
        if v > 100:
            print("Molto grande:", v)

    return totale


def main():
    x = [5, 12, 8, 15]
    y = [6, 4, 7, 2]
    incremento = 10

    risultati = elabora_elementi(x, y, incremento)
    stampa_valori(risultati)

    totale = somma_valori(1, 2, 3, 4)
    print("Somma:", totale)

    totale_lista = analizza_lista(risultati)
    print("Totale analizzato:", totale_lista)


if __name__ == "__main__":
    main()