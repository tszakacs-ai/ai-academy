import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Carica i dati
df = pd.read_csv("G2E2_clienti_churn_molti_dati.csv")
X = df[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y = df["churn"]

# Suddividi train/test (qui esempio: primi 45 train, ultimi 15 test)
X_train = X.iloc[:45]
y_train = y.iloc[:45]
X_test = X.iloc[45:]
y_test = y.iloc[45:]

# Addestra il modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calcola le probabilità per la classe "1" (churn)
y_prob = model.predict_proba(X_test)[:, 1]

# Calcola ROC e AUC
fpr, tpr, soglie = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Visualizza la curva ROC
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Modello casuale")
plt.xlabel("Tasso di falsi positivi (FPR)")
plt.ylabel("Tasso di veri positivi (TPR)")
plt.title("Curva ROC - Churn prediction")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
