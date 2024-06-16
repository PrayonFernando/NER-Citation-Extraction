
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os

# 1. Read the CSV files into Pandas DataFrames
train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')

# 2. Show the first 5 rows of each DataFrame
print("Training Set:")
print(train_df.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\nTesting Set:")
print(test_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# 3. Get information about the columns in each DataFrame
print("\nTraining Set Info:")
print(train_df.info())

print("\nTesting Set Info:")
print(test_df.info())

# 4. Create a new column called `label` in both DataFrames
train_df['label'] = train_df['Official Citation'].notnull().astype(int)
test_df['label'] = test_df['Official Citation'].notnull().astype(int)

# 5. Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# 6. Fit and transform the texts into a TF-IDF matrix
X_train = vectorizer.fit_transform(train_df['Case Name'])

# 7. Create a LogisticRegression classifier object
clf = LogisticRegression()

# 8. Fit the logistic regression model on the TF-IDF matrix and the `label` column of the training set
clf.fit(X_train, train_df['label'])

# 9. Transform the `Case Name` column of the testing set into a TF-IDF matrix using the same vectorizer
X_test = vectorizer.transform(test_df['Case Name'])

# 10. Predict the labels for the testing set using the trained model
y_pred = clf.predict(X_test)

# 11. Evaluate the model's performance
precision = classification_report(test_df['label'], y_pred, target_names=['No Citation', 'Citation'], output_dict=True)['Citation']['precision']
recall = classification_report(test_df['label'], y_pred, target_names=['No Citation', 'Citation'], output_dict=True)['Citation']['recall']
f1 = classification_report(test_df['label'], y_pred, target_names=['No Citation', 'Citation'], output_dict=True)['Citation']['f1-score']

print("\nModel Performance:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 12. Save the trained model
os.makedirs('models', exist_ok=True)  # Create the models directory if it doesn't exist
pickle.dump(clf, open('models/citation_classifier.pkl', 'wb'))
