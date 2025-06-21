import numpy as np
import pandas as pd

# Generazione dati
np.random.seed(42)
numero_dati = 50
mq = np.random.randint(40, 150, numero_dati)
stanze = np.random.randint(1, 6, numero_dati)
distanza = np.round(np.random.uniform(0.5, 10, numero_dati), 2)
prezzo = mq * 1200 + stanze * 15000 - distanza * 2000 + np.random.normal(0, 10000, numero_dati)

df = pd.DataFrame({
    "mq": mq,
    "stanze": stanze,
    "distanza_centro": distanza,
    "prezzo": prezzo.astype(int)
})

# Stampa anteprima
print(df.head())

# Salva in CSV
df.to_csv("G2E1_case_dataset.csv", index=False)
print("Dataset salvato in 'G2E1_case_dataset.csv'")
