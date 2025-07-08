import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Load the dataset
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'])
df.head()

# # Preprocess the data
# df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#         df['message'], df['label'], test_size=0.2, random_state=42
#         )

# # Preprocessing of the messages using CountVectorizer
# vectorizer = CountVectorizer()
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

# # Initialize the Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  

# # Train the model with timing
# print("Training the model...")
# start_time = time.time()
# rf_classifier.fit(X_train_vectorized, y_train)
# end_time = time.time() 
# print(f"Model trained in {end_time - start_time:.2f} seconds.")

# # Model evaluation
# print("Evaluating the model...")
# y_pred = rf_classifier.predict(X_test_vectorized)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# # Test the model with a sample message
# sample_message = ["Congratulations! You've won a free ticket to Bahamas! Call now!",
#                   "Hey, how are you doing today? Let's catch up later.",
#                   "You have a new message from your bank. Please check your account.",
#                   "Limited time offer! Get a 50% discount on your next purchase."
#                   ]
# sample_vectorized = vectorizer.transform(sample_message)
# sample_predictions = rf_classifier.predict(sample_vectorized)

# # Display the predictions for the sample messages
# for message, prediction in zip(sample_message, sample_predictions):
#     label = 'spam' if prediction == 1 else 'ham'
#     print(f"Message: {message}\nPredicted label: {label}\n")

