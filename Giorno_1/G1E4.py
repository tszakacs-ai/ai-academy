def compute_values(list_x, list_y, increment):
    results = []

    for x, y in zip(list_x, list_y):
        if x > 10:
            result = x + y + increment
        elif y > 5:
            result = x * 2 + y
        else:
            result = 0
        results.append(result)
    
    return results

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

# codice inutilizzato
# def mai_usata():
#     pass

print("Fine programma")
