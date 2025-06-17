def loop(x, y, z, temp:list):
    for i in range(len(x)):
        if x[i] > 10:
            temp.append(x[i] + y[i] + z)
        else:
            if y[i] > 5:
                temp.append(x[i] * 2 + y[i])
            else:
                temp.append(0)
    return temp

def data(x, y, z):
    return loop(x, y, z)

def foo(a, b, c, d):
    if a > 0 and b > 0 and c > 0 and d > 0:
        print("Tutti positivi")
    return a+b+c+d

def funzione_che_esamina_i_numeri(lista):
    total = 0
    for el in lista:
        toal += el if el >0 else -el
        print("Pari:" if el % 2 == 0 else "Dispari:", el)
        print("Elemento:", el)
        if el > 100:
            print("Molto grande:", el)
    return total

print("Fine programma")
