def data(x, y, z):
    """
    Elabora due liste di numeri applicando condizioni specifiche e stampa i risultati.
    
    Parameters
    ----------
    x : list
        Lista di numeri da elaborare come primo operando.
    y : list
        Lista di numeri da elaborare come secondo operando.
        Deve avere la stessa lunghezza di x.
    
    Returns
    -------
    list
        Lista contenente i valori elaborati secondo le seguenti regole:
        - Se x[i] > 10: x[i] + y[i] + z (nota: z deve essere definito globalmente)
        - Altrimenti, se y[i] > 5: x[i] * 2 + y[i]
        - Altrimenti: 0
    """
    temp = []
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
    return temp


def foo(a, b, c, d):
    """
    Somma quattro numeri e verifica se i primi tre sono tutti positivi.
    
    Parameters
    ----------
    a : float
        Primo numero da sommare.
    b : float
        Secondo numero da sommare.
    c : float
        Terzo numero da sommare.
    d : float
        Quarto numero da sommare.
    
    Returns
    -------
    float
        La somma di tutti e quattro i parametri (a + b + c + d).
    """
    if a > 0 and b > 0 and c > 0:
        print("Tutti positivi")
    return a+b+c+d


def funzione_che_esamina_i_numeri(lista):
    """
    Analizza una lista di numeri calcolando un totale modificato e stampando informazioni dettagliate.
    
    Parameters
    ----------
    lista : list
        Lista di numeri da analizzare.
    
    Returns
    -------
    float or int
        Totale calcolato secondo le seguenti regole:
        - Se l'elemento è positivo: viene aggiunto al totale
        - Se l'elemento è negativo o zero: viene sottratto dal totale (in valore assoluto)
    """
    total = 0
    for el in lista:
        print("Elemento:", el)
        if el > 0:
            total += el
        else:
            total -= el
        if el % 2 == 0:
            print("Pari:", el)
        else:
            print("Dispari:", el)
        if el > 100:
            print("Molto grande:", el)
    result = total
    return result

def main():
    data([1, 2, 3, 11, 12], [6, 7, 8, 9, 10], 5)
    foo(1, 2, 3, 4)
    funzione_che_esamina_i_numeri([1, 2, 3, -4, -5, 6, 7, 8, 9, 10])
