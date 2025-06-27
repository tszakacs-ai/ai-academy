def crealista(x, y, z):
        
    """
    Calcola una lista di valori in base a partire da x e y.

    Parameters
    ----------
    x : list of int or float
        Lista dei primi valori di input.
    y : list of int or float
        Lista dei secondi valori di input.
    z : int or float
        Valore costante da sommare in certe condizioni.

    Returns
    -------
    list of int or float
        Lista dei valori calcolati.
    """
        
    temp = []

    for x, y in zip(x, y):
        if x > 10:
            temp.append(x + y + z)
        elif y > 5:
            temp.append(x * 2 + y)
        else:
            temp.append(0)
        return temp

    return temp


def stampa_analisi_lista(temp):

    """
    Stampa messaggi specifici in base al valore degli elementi della lista.

    Parameters
    ----------
    temp : list of int or float
        Lista di valori.
    """
    for x in temp:
        if x > 50:
            print("Grande valore trovato!")
        elif x == 0:
            print("Zero trovato!")
        else:
            print("Valore: ", x)


def positivi(a, b, c, d):

    """
    stampa se i numeri inseriti sono tutti positivi e la loro somma.

    Parameters
    ----------
    a,b,c,d : int or float
        numeri da sommare

    Returns
    -------
    int or float
    La somma dei numeri a, b, c e d.
    """

    if a > 0 and b > 0 and c > 0 and  d > 0:
        print("Tutti positivi")
    return a+b+c+d

def somma_lista(lista):
    """
    Calcola la somma degli elementi di una lista, aggiungendo gli elementi positivi e sottraendo quelli negativi.
    Parameters
    ----------
    lista : list of int or float
        Lista di numeri 

    Returns
    -------
    int or float
    """
    somma = 0
    for el in lista:
        if el > 0:
            somma += el
        else:
            somma -= el
    return somma


def report_lista(lista):
    """
    Stampa vari report sugli elementi di una lista, come se sono pari o dispari, se sono molto grandi, e li stampa uno per uno.
    Parameters
    ----------
    lista : list of int or float
        Lista di numeri analizzati.

    """
    for el in lista:
        if el % 2 == 0:
            print("Pari:", el)
        else:
            print("Dispari:", el)

    for el in lista:
        print("Elemento:", el)

    for el in lista:
        if el > 100:
            print("Molto grande:", el)


def main():
    # Esempio di esecuzione del programma completo
    x = [1, 2, 3, 4]
    y = [6, 14, 7, 22]
    z = 10

    valori = crealista(x, y, z)  
    stampa_analisi_lista(valori)                    
    print("Tutti positivi?", positivi(*x, *y, z))
    totale = somma_lista(valori)
    print("Totale:", totale)
    report_lista(valori)


if __name__ == "__main__":
    main()

print("Fine programma")
