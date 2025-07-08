import pandas as pd
import matplotlib.pyplot as plt

# Carica i dati originali e le predizioni
output = pd.read_csv("G2E1_risultati_regressione.csv")

plt.figure(figsize=(8,6))

# Dati originali: prezzo reale (X) vs prezzo reale (Y) – diagonale
plt.scatter(output["prezzo"], output["prezzo"], color='blue', alpha=0.5, label="Dati reali (perfetti)")

# Predizioni: prezzo reale (X) vs prezzo predetto (Y)
plt.scatter(output["prezzo"], output["prezzo_predetto"], color='green', label="Predizioni modello")

# Linea ideale (perfetta)
plt.plot([output["prezzo"].min(), output["prezzo"].max()],
         [output["prezzo"].min(), output["prezzo"].max()],
         color='red', linestyle='--', label="Linea ideale (perfetta)")

plt.xlabel("Prezzo reale")
plt.ylabel("Prezzo predetto")
plt.title("Prezzo reale vs prezzo predetto: dati, predizioni e linea ideale")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
