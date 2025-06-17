import numpy as np
import pandas as pd

np.random.seed(101)
numero_dati = 60

# Genera feature
num_reclami = np.random.randint(0, 6, numero_dati)
mesi_ultimo_acquisto = np.random.randint(0, 25, numero_dati)
spesa_mensile = np.random.randint(10, 300, numero_dati)
cliente_vip = np.random.binomial(1, 0.2, numero_dati)

# Churn: combinazione di regole + rumore
# Churn se almeno due condizioni di rischio vere, oppure una ma con "forte rumore"
condizioni_rischio = (
    (num_reclami > 3).astype(int) +
    (mesi_ultimo_acquisto > 12).astype(int) +
    (spesa_mensile < 50).astype(int)
)

# Base: churn per chi ha almeno 2 condizioni
churn = np.where(condizioni_rischio >= 2, 1, 0)

# Per rendere la classificazione meno banale:
# aggiungi un po' di churn random su clienti normali (per simulare incertezza reale)
extra_churn = (np.random.rand(numero_dati) < 0.12).astype(int)
churn = np.maximum(churn, extra_churn)

# Assicurati che il churn sia almeno il 25%
while churn.sum() < numero_dati * 0.25:
    idx = np.random.choice(np.where(churn == 0)[0])
    churn[idx] = 1

df_churn = pd.DataFrame({
    "num_reclami": num_reclami,
    "mesi_ultimo_acquisto": mesi_ultimo_acquisto,
    "spesa_mensile": spesa_mensile,
    "cliente_vip": cliente_vip,
    "churn": churn
})

print("Distribuzione churn:\n", df_churn["churn"].value_counts(normalize=True))
df_churn.to_csv("G2E1_clienti_churn.csv", index=False)
print("Dataset clienti salvato in 'G2E1_clienti_churn.csv'")
