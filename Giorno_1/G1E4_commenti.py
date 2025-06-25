# ESEMPIO DI CODICE "SPORCO" DA RIPULIRE
# (ogni punto evidenzia un problema visto in aula e quale tecnica va usata per correggerlo)

def preprocessing(lista_x, lista_y, incremento):
    """
    Calcola una lista di valori in base a regole definite sugli elementi di lista_x e lista_y.

    Parameters
    ----------
    lista_x : list of int or float
        Lista dei primi valori di input.
    lista_y : list of int or float
        Lista dei secondi valori di input (stessa lunghezza di lista_x).
    incremento : int or float
        Valore costante da sommare in certe condizioni.

    Returns
    -------
    list of int or float
        Lista dei valori calcolati.
    """
    risultati = []
    for x, y in zip(lista_x, lista_y):
        if x > 10:
            risultato = x + y + incremento
        elif y > 5:
            risultato = x * 2 + y
        else:
            risultato = 0
        risultati.append(risultato)
    return risultati

def controllo_valori(valori):
    """
    Analizza e stampa messaggi specifici sui valori calcolati.

    Parameters
    ----------
    valori : list of int or float
        Lista di valori su cui effettuare l'analisi.

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

def controllo_valori_positivi(*args):
    """
    Verifica se tutti i valori forniti sono positivi.

    Parameters
    ----------
    *args : int or float
        Qualsiasi numero di argomenti numerici.

    Returns
    -------
    bool
        True se tutti i valori sono positivi, False altrimenti.
    """
    return all(a > 0 for a in args)

def calcola_totale(lista):
    """
    Calcola il totale di una lista sommando i positivi e sottraendo i negativi.

    Parameters
    ----------
    lista : list of int or float
        Lista dei valori.

    Returns
    -------
    int or float
        Totale calcolato.
    """
    totale = 0
    for el in lista:
        if el > 0:
            totale += el
        else:
            totale -= el
    return totale

def controllo_pari_dispari(lista):
    """
    Stampa se gli elementi della lista sono pari o dispari.

    Parameters
    ----------
    lista : list of int or float
        Lista di valori.

    Returns
    -------
    None
    """
    for el in lista:
        if el % 2 == 0:
            print("Pari:", el)
        else:
            print("Dispari:", el)

def visualizza_elementi(lista):
    """
    Stampa tutti gli elementi della lista.

    Parameters
    ----------
    lista : list of int or float
        Lista di valori.

    Returns
    -------
    None
    """
    for el in lista:
        print("Elemento:", el)

def visualizza_grandi_valori(lista, soglia=100):
    """
    Stampa i valori della lista che superano una soglia.

    Parameters
    ----------
    lista : list of int or float
        Lista di valori.
    soglia : int or float, optional
        Valore sopra il quale considerare un elemento grande (default 100).

    Returns
    -------
    None
    """
    for el in lista:
        if el > soglia:
            print("Molto grande:", el)

def stampa_report_completo(lista):
    """
    Stampa un report completo sulla lista.

    Parameters
    ----------
    lista : list of int or float
        Lista di valori.

    Returns
    -------
    None
    """
    controllo_pari_dispari(lista)
    visualizza_elementi(lista)
    visualizza_grandi_valori(lista)

def main():
    """
    Funzione principale del programma.
    """
    x = [5, 12, 8, 15]
    y = [6, 4, 7, 2]
    incremento = 10

    valori = preprocessing(x, y, incremento)
    controllo_valori(valori)
    print("Tutti positivi?", controllo_valori_positivi(*x, *y, incremento))
    totale = calcola_totale(valori)
    print("Totale:", totale)
    stampa_report_completo(valori)
    print("Fine programma")

if __name__ == "__main__":
    main()