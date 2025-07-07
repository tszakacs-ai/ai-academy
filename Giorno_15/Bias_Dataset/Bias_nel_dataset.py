import pandas as pd

# Definizione delle colonne
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", 
    "income"
]

# Caricamento del dataset locale
df = pd.read_csv("./Giorno_15/Bias_Dataset/adult.csv", header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Analisi delle colonne sensibili
print(df["sex"].value_counts())
print(df["race"].value_counts())
