import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Carica dati
df = pd.read_csv("G2E2_clienti_churn_molti_dati.csv")
X = df[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y = df["churn"]

# Imposta K-fold stratificato (es: 5 fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
aucs = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilità della classe "1"
    
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float('nan')  # In caso di una sola classe in un fold
    
    accuracies.append(acc)
    aucs.append(auc)

print("Accuracy media sui 5 fold: {:.2f}".format(sum(accuracies)/len(accuracies)))
print("AUC media sui 5 fold: {:.2f}".format(sum([a for a in aucs if not pd.isna(a)])/len([a for a in aucs if not pd.isna(a)])))
