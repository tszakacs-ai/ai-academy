import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Carica i dati
df = pd.read_csv("G2E2_clienti_churn_molti_dati.csv")
X = df[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y = df["churn"]

# Train/test split stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Allena il modello
model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, 
                      use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

# Predizioni
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Estrai TP, FP, TN, FN dalla confusion matrix
# [[TN, FP],
#  [FN, TP]]
TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)  # Safety for binary

# Metriche "a mano"
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Altre metriche sklearn
accuracy_sk = accuracy_score(y_test, y_pred)
precision_sk = precision_score(y_test, y_pred, pos_label=1)
recall_sk = recall_score(y_test, y_pred, pos_label=1)
f1_sk = f1_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_prob)

print("\n--- Metriche calcolate manualmente ---")
print(f"Accuracy = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN}) = {(TP + TN)} / {(TP + TN + FP + FN)} = {accuracy:.2f} ({accuracy*100:.0f}%)")
print(f"Precision = {TP} / ({TP} + {FP}) = {TP} / {TP + FP} = {precision:.2f} ({precision*100:.0f}%)")
print(f"Recall = {TP} / ({TP} + {FN}) = {TP} / {TP + FN} = {recall:.2f} ({recall*100:.0f}%)")
print(f"F1 = 2*({precision:.2f}*{recall:.2f}) / ({precision:.2f}+{recall:.2f}) = {f1:.2f} ({f1*100:.0f}%)")

print("\n--- Metriche sklearn ---")
print(f"Accuracy: {accuracy_sk:.2f}")
print(f"Precision: {precision_sk:.2f}")
print(f"Recall: {recall_sk:.2f}")
print(f"F1-score: {f1_sk:.2f}")
print(f"AUC: {auc:.2f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
