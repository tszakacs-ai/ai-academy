# Il codice sporco da ripulire


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
        if temp[i] > 50:
            print("Grande valore trovato!")
        else:
            if temp[i] == 0:
                print("Zero trovato!")
            else:
                print("Valore: ", temp[i])

    return temp

def foo(a, b, c, d):
    if a > 0 and b > 0 and c > 0:
        print("Tutti positivi")
    return a+b+c+d

def funzione_che_esamina_i_numeri(lista):
    total = 0
    for el in lista:
        if el > 0:
            total += el
        else:
            total -= el

        if el % 2 == 0:
            print("Pari:", el)
        else:
            print("Dispari:", el)

        print("Elemento:", el)

        if el > 100:
            print("Molto grande:", el)
    
    result = total
    return result


print("Fine programma")
