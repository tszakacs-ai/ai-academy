from NER_Graph import normalizza_nomi

def test_normalizza():
    nomi = ["Mario Rossi", "M. Rossi", "Luigi Bianchi", "L. Bianchi", "Anna Verdi"]
    codici = normalizza_nomi(nomi, soglia=0.7)
    print(codici)

if __name__ == "__main__":
    test_normalizza()