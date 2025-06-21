import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("G2E1_clienti_churn.csv")
train = df.iloc[:45]
test = df.iloc[45:]

X_train = train[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_train = train["churn"]
X_test = test[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_test = test["churn"]

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Valutazione
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Salva predizioni
output = test.copy()
output["churn_predetto"] = y_pred
output.to_csv("G2E1_risultati_churn.csv", index=False)
print("Predizioni salvate in 'G2E1_risultati_churn.csv'")
