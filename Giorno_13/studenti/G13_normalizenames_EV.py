from NER_Graph import normalizza_nomi

def prova_normalizzazione():
    lista_nomi = ["Marco Verdi", "M. Verdi", "Mario Rossi", "M. Rossi", "Giulia Neri"]
    gruppi_nomi = normalizza_nomi(lista_nomi, soglia=0.7)
    print(gruppi_nomi)

if __name__ == "__main__":
    prova_normalizzazione()