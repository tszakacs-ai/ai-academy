import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Carica dati
df = pd.read_csv("G2E2_clienti_churn_molti_dati.csv")
X = df[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
    "use_label_encoder": [False],
    "eval_metric": ["logloss"],
    "random_state": [42]
}

xgb = XGBClassifier()

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=2,           # Con pochi dati usa cv=2
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("Migliori parametri trovati:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"AUC finale sul test set: {auc:.2f}")
print(f"Accuracy finale sul test set: {acc:.2f}")
print("Confusion matrix:\n", cm)
