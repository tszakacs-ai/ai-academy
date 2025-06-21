import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Leggi il dataset dal file CSV
df = pd.read_csv("G2E1_case_dataset.csv")

# Split train/test (ad esempio: primi 40 train, ultimi 10 test)
train = df.iloc[:40]
test = df.iloc[40:]

X_train = train[["mq", "stanze", "distanza_centro"]]
y_train = train["prezzo"]
X_test = test[["mq", "stanze", "distanza_centro"]]
y_test = test["prezzo"]

# Modello e training
model = LinearRegression()
model.fit(X_train, y_train)

# Predizione sul test set
y_pred = model.predict(X_test)

# Valutazione e stampa delle metriche
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))

# Salva predizioni su file
output = test.copy()
output["prezzo_predetto"] = y_pred.astype(int)
output.to_csv("G2E1_risultati_regressione.csv", index=False)
print("Predizioni salvate in 'G2E1_risultati_regressione.csv'")
