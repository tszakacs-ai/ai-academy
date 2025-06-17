# Codice pulito

def analizza_valori_lista(x, y, z):
    """
    Analizza i valori generati da una lista combinata e stampa messaggi in base al valore.

    Args:
        x (list of int/float): Prima lista di numeri.
        y (list of int/float): Seconda lista di numeri.
        z (int/float): Valore da combinare.

    Returns:
        list: Lista dei valori combinati.
    """
    temp = genera_lista_combinata(x, y, z)
    for val in temp:
        if val > 50:
            print("Grande valore trovato!")
        elif val == 0:
            print("Zero trovato!")
        else:
            print("Valore:", val)
    return temp

def genera_lista_combinata(x, y, z):
    """
    Genera una lista combinata in base a condizioni sui valori delle liste x e y.

    Args:
        x (list of int/float): Prima lista di numeri.
        y (list of int/float): Seconda lista di numeri.
        z (int/float): Valore da combinare.

    Returns:
        list: Lista risultante dalla combinazione.
    """
    temp = []
    for i in range(len(x)):
        if x[i] > 10:
            temp.append(x[i] + y[i] + z)
        elif y[i] > 5:
            temp.append(x[i] * 2 + y[i])
        else:
            temp.append(0)
    return temp

def somma_e_controlla_positivi(a, b, c, d):
    """
    Somma quattro numeri e stampa se sono tutti positivi.

    Args:
        a, b, c, d (int/float): Numeri da sommare.

    Returns:
        int/float: La somma dei quattro numeri.
    """
    if a > 0 and b > 0 and c > 0 and d > 0:
        print("Tutti positivi.")
    else:
        print("Non tutti positivi.")
    return a + b + c + d

def analizza_e_stampa_info_numeri(lista):
    """
    Analizza una lista di numeri, stampa informazioni su pari/dispari e grandezza, e calcola un totale.

    Args:
        lista (list of int/float): Lista di numeri.

    Returns:
        int/float: Il totale calcolato.
    """
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
    result = total
    return result

def main():
    """
    Esempio di utilizzo delle funzioni definite nel file.
    """
    x = [12, 5, 8]
    y = [7, 3, 10]
    incremento = 4
    print("Analisi valori lista:")
    analizza_valori_lista(x, y, incremento)

    print("Somma e controllo positivi:")
    somma_e_controlla_positivi(1, 2, -4, 4)

    print("Analizza e stampa info numeri:")
    analizza_e_stampa_info_numeri([10, -5, 102, 3])

    print("Fine programma")
    
if __name__ == "__main__":
    main()

