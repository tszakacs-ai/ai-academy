# Il codice sporco da ripulire

global_value = 0 # variabile globale usata a caso

def data(x, y, z):
    temp = []
    # codice morto
    # print("Questo era un vecchio debug")

    for i in range(len(x)):
        if x[i] > 10:
            temp.append(x[i] + y[i] + z)
        else:
            if y[i] > 5:
                temp.append(x[i] * 2 + y[i])
            else:
                temp.append(0)
    for i in range(len(x)):
        if x[i] > 10:
            temp.append(x[i] + y[i] + z)
        else:
            if y[i] > 5:
                temp.append(x[i] * 2 + y[i])
            else:
                temp.append(0)

    for i in range(len(x)):
        if temp[i] > 50:
            print("Grande valore trovato!")
        else:
            if temp[i] == 0:
                print("Zero trovato!")
            else:
                print("Valore: ", temp[i])

    useless = 5  # parametro inutilizzato
    return temp

def foo(a, b, c, d):
    if a > 0:
        if b > 0:
            if c > 0:
                print("Tutti positivi")
    return a+b+c+d

def funzione_che_esamina_i_numeri(lista):
    total = 0
    for el in lista:
        if el > 0:
            total += el
        else:
            total -= el
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
    # codice morto
    # return 123
    result = total
    return result

def calcola_valori(lista_x, lista_y, incremento):
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

def stampa_analisi(valori):
    """
    Analizza e stampa messaggi specifici sui valori calcolati.

    Parameters
    ----------
    valori : list of int or float
        Lista di valori su cui effettuare l’analisi.

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

def tutti_positivi(*args):
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

def stampa_report(lista):
    """
    Stampa varie analisi sugli elementi della lista.

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
    for el in lista:
        print("Elemento:", el)
    for el in lista:
        if el > 100:
            print("Molto grande:", el)

def main():
    """
    Funzione principale per eseguire il programma.
    """
    x = [5, 12, 8, 15]
    y = [6, 4, 7, 2]
    incremento = 10

    valori = calcola_valori(x, y, incremento)  # Logica principale, funzione breve e chiara
    stampa_analisi(valori)                    # Separazione output
    print("Tutti positivi?", tutti_positivi(*x, *y, incremento))
    totale = calcola_totale(valori)
    print("Totale:", totale)
    stampa_report(valori)
    print("Programma completato con successo.")  # Messaggio finale chiaro

if __name__ == "__main__":
    main()

print("Fine programma")