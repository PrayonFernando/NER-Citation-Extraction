import json
import os
import pickle
import spacy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.models.citation_model import train_and_evaluate_model, preprocess_text
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
import logging

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Define label2id dictionary
label_to_id = {'O': 0, 'B-CITATION': 1, 'I-CITATION': 2}


def extract_features(tokens, citation_classifier, tfidf_vectorizer, max_seq_length):
    """Extracts features for a sequence of tokens and returns a list of lists of features."""
    features = []
    for i, token in enumerate(tokens):
        token_features = []
        # POS Tag
        doc = nlp(token)
        token_features.append(doc[0].pos_)

        # Character Features
        token_features.append(token[0].isupper())
        token_features.append(token.isdigit())

        # Get predictions from citation classifier only for non-padded tokens
        if token != 'O':
            text_tfidf = tfidf_vectorizer.transform([token])
            prob = citation_classifier.predict_proba(text_tfidf)[0][1]
        else:
            prob = 0.0  # Set probability to 0 for padded tokens

        token_features.append(prob)  # Add citation probability as a feature
        features.append(token_features)

    # Pad features to ensure consistent sequence length (use 0 for numeric padding)
    for i in range(len(features), max_seq_length):
        features.append(['', 0, 0, 0.0])

    return features


# Load your annotated training and testing data
with open("../data/processed/annotated_data/train_data.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]
with open("../data/processed/annotated_data/test_data.jsonl", "r") as f:
    test_data = [json.loads(line) for line in f]

# Train and evaluate the citation model
citation_classifier, tfidf_vectorizer = train_and_evaluate_model()  # This will train the classifier and return it


max_seq_length = max(len(sample["tokens"]) for sample in train_data)


def pad_sequences(sequences, max_length, padding_value='O'):
    """Pads sequences to a specified length with the given padding value."""
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_sequences.append(seq + [padding_value] * (max_length - len(seq)))
        else:
            padded_sequences.append(seq[:max_length])  # Truncate if longer
    return padded_sequences


def create_label_to_id(labels):
    """Creates a mapping from labels to numerical indices."""
    unique_labels = set(labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    return label_to_id


def encode_labels(sequences, label_to_id):
    """Encodes string labels to numerical indices based on a label-to-id mapping."""
    encoded_sequences = []
    for seq in sequences:
        encoded_sequences.append([label_to_id[label] for label in seq])
    return encoded_sequences


# Extract features for training and testing data
X_train = [extract_features(sample["tokens"], citation_classifier, tfidf_vectorizer, max_seq_length) for sample in
           train_data]
X_test = [extract_features(sample["tokens"], citation_classifier, tfidf_vectorizer, max_seq_length) for sample in
          test_data]

# Extract labels
y_train = [sample["labels"] for sample in train_data]
y_test = [sample["labels"] for sample in test_data]

# Pad sequences
y_train_padded = pad_sequences(y_train, max_seq_length)
y_test_padded = pad_sequences(y_test, max_seq_length)

# Combine all labels after padding
all_labels = [label for seq in y_train_padded for label in seq] + [label for seq in y_test_padded for label in seq]
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Save the updated label encoder
with open('../models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Encode labels
y_train_encoded = encode_labels(y_train_padded, label_to_id)
y_test_encoded = encode_labels(y_test_padded, label_to_id)
# ------------------MODEL TRAINING-----------------
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train_encoded)

# Evaluate the model
y_pred = crf.predict(X_test)

# Flatten the predictions
y_pred_flat = [label for seq in y_pred for label in seq]
y_test_flat = [label for seq in y_test_padded for label in seq]

# Calculate and print the classification report (using sklearn)
from sklearn.metrics import classification_report

print(classification_report(
    y_test_flat,
    y_pred_flat,
    labels=list(label_encoder.classes_),
    digits=3
))

joblib.dump(crf, '../models/citation_extractor_crf.pkl')  # Save the model
