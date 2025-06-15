# ESEMPIO DI CODICE "SPORCO" DA RIPULIRE
# (ogni punto evidenzia un problema visto in aula e quale tecnica va usata per correggerlo)

global_value = 0  # [DA EVITARE] Variabile globale non necessaria (preferire variabili locali o parametri)

def data(x, y, z):
    # [CODE SMELL: NOME GENERICO]
    # Rinominare la funzione con un nome chiaro che spiega cosa fa
    temp = []
    
    # [CODICE MORTO]
    # Eliminare o sostituire print di debug/commentate
    # print("Questo era un vecchio debug")
    
    # [CODE SMELL: DUPLICAZIONE]
    # Il ciclo seguente è duplicato (stessa logica due volte di fila). Estrarre in una funzione separata.
    for i in range(len(x)):
        # [STRUTTURE ANNIDATE]
        # Troppa annidamento. Ristrutturare usando funzioni e if più piatti.
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
    
    # [CODE SMELL: LOGICA ANNIDATA E STAMPA DIRETTA]
    # Separare la logica di controllo e la stampa in funzioni dedicate, evitare stampe in funzioni "core"
    for i in range(len(x)):
        if temp[i] > 50:
            print("Grande valore trovato!")  # Preferire return/log rispetto a print diretto
        else:
            if temp[i] == 0:
                print("Zero trovato!")
            else:
                print("Valore: ", temp[i])
    
    useless = 5  # [PARAMETRO/VARIABILE INUTILE] - da eliminare
    
    return temp

def foo(a, b, c, d):
    # [NOME GENERICO, STRUTTURE ANNIDATE]
    # Rinominare la funzione in modo significativo e ridurre l'annidamento.
    if a > 0:
        if b > 0:
            if c > 0:
                print("Tutti positivi")
    return a+b+c+d

def funzione_lunghissima(lista):
    # [FUNZIONE TROPPO LUNGA]
    # Spezzare questa funzione in più sotto-funzioni
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
    # [CODICE MORTO]
    # Eliminare codice non usato
    # return 123
    result = total
    return result

# [CODICE INUTILE/NON USATO]
# Definire solo funzioni effettivamente utilizzate nel progetto
# def mai_usata():
#     pass


"""
Esempio di refactoring di una code-base "sporca":
- Eliminazione variabili globali
- Uso di nomi chiari e funzioni brevi
- Rimozione duplicazione e codice morto
- Semplificazione strutture annidate
- Separazione logica/calcolo da input/output
- Docstring NumPy-style
"""

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

# Mancano dati esempi e l'avvio del programma

def main():
    # Esempio di esecuzione del programma completo
    x = [5, 12, 8, 15]
    y = [6, 4, 7, 2]
    incremento = 10

    valori = calcola_valori(x, y, incremento)  # Logica principale, funzione breve e chiara
    stampa_analisi(valori)                    # Separazione output
    print("Tutti positivi?", tutti_positivi(*x, *y, incremento))
    totale = calcola_totale(valori)
    print("Totale:", totale)
    stampa_report(valori)

if __name__ == "__main__":
    main()


print("Fine programma")  # [STAMPA NON MOTIVATA] – da eliminare o spostare in main/log

