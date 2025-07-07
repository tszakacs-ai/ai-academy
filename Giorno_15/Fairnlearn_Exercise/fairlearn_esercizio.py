import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from fairlearn.metrics import MetricFrame, selection_rate
import numpy as np

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", 
    "income"
]

# Caricamento del dataset
data = pd.read_csv(
    "./Giorno_15/Bias_Dataset/adult.csv",
    header=None,
    names=columns,
    na_values=" ?",
    skipinitialspace=True
)

# Pulizia dei dati
data = data.dropna()

# Feature e target
X = data.drop(columns=["income"])
y = data["income"].apply(lambda x: 1 if x == ">50K" else 0)

# Suddivisione in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Colonne categoriche da codificare
categorical_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Addestramento del modello
model = LinearRegression()
model.fit(X_train_encoded, y_train)

# Predizione
predictions = model.predict(X_test_encoded)
y_pred = np.round(predictions)

# Calcolo della Demographic Parity rispetto al genere
sex_test = X_test["sex"]
mf = MetricFrame(
    metrics=selection_rate,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sex_test
)

# Output delle metriche
print("\nDemographic Parity (selection rate) per gruppo 'sex':")
print(mf.by_group)
