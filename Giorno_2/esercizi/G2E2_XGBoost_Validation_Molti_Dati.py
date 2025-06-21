import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carica dati
df = pd.read_csv("G2E2_clienti_churn_molti_dati.csv")

# Split: primi 35 = train, successivi 10 = validation, ultimi 15 = test
train = df.iloc[:35]
val = df.iloc[35:45]
test = df.iloc[45:]

X_train = train[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_train = train["churn"]
X_val = val[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_val = val["churn"]
X_test = test[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_test = test["churn"]

# Addestra modello XGBoost con tuning
model_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.01,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Predizioni sul test set
y_pred = model_xgb.predict(X_test)

# Tabella risultati
output = test
output["churn_predetto_xgb"] = y_pred
output.to_csv("G2E2_risultati_churn_xgb_validation_molti_dati.csv", index=False)
print("Risultati salvati in 'G2E2_risultati_churn_xgb_validation_molti_dati.csv'")

# Metriche valutazione su test set
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
