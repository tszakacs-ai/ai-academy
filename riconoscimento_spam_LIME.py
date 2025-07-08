import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_text
import numpy as np

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# converti in minuscolo
df['text'] = df['text'].str.lower()

# dividi dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# vettorizzazione
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#modello Naive Bayes e random forest
model = MultinomialNB()
model.fit(X_train_vec, y_train)

model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train_vec, y_train)

# Predizione e valutazione
y_pred = model.predict(X_test_vec)
print("Naive Bayes Classification Report:")
print("accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

y_pred2 = model2.predict(X_test_vec)
print("Random Forest Classification Report:")
print("accuracy:", accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

# --- LIME Explanation ---

# Create a LIME explainer
class_names = model2.classes_ # ['ham', 'spam']
explainer = lime.lime_text.LimeTextExplainer(
    class_names=class_names
)

# Define a prediction function for LIME
# LIME expects a function that takes a list of raw texts and returns a numpy array
# of prediction probabilities for each class.
def predict_proba_for_lime(texts):
    # Transform the raw text using the *trained* vectorizer
    texts_vec = vectorizer.transform(texts)
    # Return the probabilities from the Naive Bayes model
    # You can choose either model (model or model2) here. Let's use model (Naive Bayes) for now.
    return model2.predict_proba(texts_vec)

print("\n--- LIME Explanations Random Forest ---")

# Select 5 random elements from the test set
np.random.seed(58) # for reproducibility
random_indices = np.random.choice(X_test.index, 5, replace=False)

for i, idx in enumerate(random_indices):
    text_to_explain = X_test.loc[idx]
    true_label = y_test.loc[idx]
    predicted_label = model.predict(vectorizer.transform([text_to_explain]))[0]

    print(f"\n--- Explanation for Sample {i+1} (Original Index: {idx}) ---")
    print(f"Text: '{text_to_explain}'")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")

    # Generate explanation for the prediction of the predicted label
    # The `predict_proba_for_lime` function is passed here.
    # `num_features` controls how many features are shown in the explanation.
    # `num_samples` controls the number of perturbed samples LIME generates.
    explanation = explainer.explain_instance(
        text_to_explain,
        predict_proba_for_lime,
        num_features=10, # Show top 10 important features
        num_samples=5000 # Number of perturbed samples for explanation
    )

    # Print the explanation (list of (word, weight) tuples)
    print("Most important features for this prediction:")
    for word, weight in explanation.as_list():
        print(f"  - {word}: {weight:.4f}")

    # Optionally, you can get the explanation for a specific class (e.g., 'spam')
    # spam_explanation = explanation.as_list(label=class_names.tolist().index('spam'))
    # print("\nFeatures contributing to 'spam' prediction:")
    # for word, weight in spam_explanation:
    #     print(f"  - {word}: {weight:.4f}")