import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fairlearn.metrics import MetricFrame

file_path = 'Giorno_15/adult/'

adult_data = pd.read_csv(file_path + 'adult.data', sep=' ', na_values=' ?', 
                     names=['age', 'workclass', 'fnlwgt', 'education',
                             'education_num', 'marital_status', 'occupation','relationship',
                             'race', 'sex', 'capital_gain', 'capital_loss',
                             'hours_per_week', 'native_country', 'income'])
# print(adult_data.head())


print(f'Count of elements in sex column:\n{adult_data['sex'].value_counts()}\n')
print(f'Count of elements in race column:\n{adult_data["race"].value_counts()}\n')

# Split the dataset into training and testing sets
X = adult_data.drop('income', axis=1)
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
y = adult_data['income'].apply(lambda x: 1 if x == '>50K' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure both training and testing sets have the same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# # Initialize the Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# # Train the model
# print("Training the model...")
# rf_classifier.fit(X_train, y_train)
# # Model evaluation
# print("Evaluating the model...")
# y_pred = rf_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# # Fairness evaluation
# metric_frame = MetricFrame(
#     metrics={
#         'accuracy': accuracy_score,
#         'false_positive_rate': lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0][1] / (confusion_matrix(y_true, y_pred)[0][1] + confusion_matrix(y_true, y_pred)[0][0]),
#         'false_negative_rate': lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1][0] / (confusion_matrix(y_true, y_pred)[1][0] + confusion_matrix(y_true, y_pred)[1][1])
#     },
#     y_true=y_test,
#     y_pred=y_pred,
#     sensitive_features=X
# )

# print("Fairness Metrics:")
# print(metric_frame.by_group)

print("Distribuzione della classe prima dell'oversampling:")
print(y_train.value_counts())
print()

 # Oversampling the minority class
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print("Distribuzione della classe dopo l'oversampling:")
print(y_resampled.value_counts())