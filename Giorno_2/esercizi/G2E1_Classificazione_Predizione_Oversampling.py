import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Carica dati e suddividi train/test
df = pd.read_csv("G2E1_clienti_churn.csv")
train = df.iloc[:45].copy()
test = df.iloc[45:].copy()

X_train = train[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_train = train["churn"]
X_test = test[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_test = test["churn"]

# SMOTE solo su training
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

print("Distribuzione train originale:\n", y_train.value_counts())
print("Distribuzione train dopo SMOTE:\n", pd.Series(y_train_bal).value_counts())
print("Distribuzione test:\n", y_test.value_counts())

# Addestra modello su dati bilanciati
model = LogisticRegression()
model.fit(X_train_bal, y_train_bal)
y_pred = model.predict(X_test)

# Tabella risultati per confronto
output = test.copy()
output["churn_predetto"] = y_pred
output.to_csv("G2E1_risultati_churn_sampling.csv", index=False)
print("Risultati salvati in 'G2E1_risultati_churn_sampling.csv'")

# Metriche per la discussione
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
