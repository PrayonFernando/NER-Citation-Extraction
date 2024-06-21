import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os
import re

def preprocess_text(text):
    """
    Preprocesses text for citation extraction:
        - Removes newlines.
        - Removes extra spaces.
        - Lowercases the text.
        - Removes any characters other than letters, numbers, spaces, and punctuation.

    Args:
        text (str): The raw text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = text.replace('\n', ' ')  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z0-9\s\.,;:\(\)]', '', text)  # Keep only alphanumeric and punctuation
    return text


def train_and_evaluate_model():
    # Define file paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, '..', '..', 'data', 'processed', 'train.csv')
    test_file = os.path.join(script_dir, '..', '..', 'data', 'processed', 'test.csv')
    model_file = os.path.join(script_dir, '..', '..', 'models', 'citation_classifier.pkl')
    vectorizer_file = os.path.join(script_dir, '..', '..', 'models', 'tfidf_vectorizer.pkl')

    # Load and preprocess training data
    train_df = pd.read_csv(train_file)
    X_train = [preprocess_text(text) for text in train_df['Case Name']]
    y_train = train_df['label'].tolist()  # Use the 'label' column

    # Load and preprocess testing data
    test_df = pd.read_csv(test_file)
    X_test = [preprocess_text(text) for text in test_df['Case Name']]
    y_test = test_df['label'].tolist()  # Use the 'label' column

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    report = classification_report(y_test, y_pred, target_names=['No Citation', 'Citation'])
    print("Model Evaluation:\n", report)

    # Save the model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel saved to {model_file}")
    print(f"Vectorizer saved to {vectorizer_file}")

    # Return the model and vectorizer
    return model, vectorizer


if __name__ == "__main__":
    train_and_evaluate_model()