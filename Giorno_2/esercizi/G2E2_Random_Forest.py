import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carica dati
df = pd.read_csv("G2E1_clienti_churn.csv")
train = df.iloc[:45]
test = df.iloc[45:]

X_train = train[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_train = train["churn"]
X_test = test[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_test = test["churn"]

# Addestra Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predizioni sul test set
y_pred = model_rf.predict(X_test)

# Tabella risultati
output = test
output["churn_predetto_rf"] = y_pred
output.to_csv("G2E2_risultati_churn_random_forest.csv", index=False)
print("Risultati salvati in 'G2E2_risultati_churn_random_forest.csv'")

# Metriche di valutazione
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
