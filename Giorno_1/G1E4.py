#Codice pulito
def analizza_valori_lista(x, y, z):
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
    if a > 0 and b > 0 and c > 0 and d > 0:
        print("Tutti positivi.")
    return a + b + c + d

def analizza_e_stampa_info_numeri(lista):
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

print("Fine programma")
