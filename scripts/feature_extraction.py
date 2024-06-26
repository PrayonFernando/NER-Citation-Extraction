import json
import os
import pickle
import spacy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.models.citation_model import train_and_evaluate_model, preprocess_text
from spacy.training.example import Example
from spacy.scorer import Scorer

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Define label_to_id outside of the functions
label_to_id = {'O': 0, 'B-CITATION': 1, 'I-CITATION': 2,
               'B-CASE_NAME': 3, 'I-CASE_NAME': 4,
               'B-COURT_NAME': 5, 'I-COURT_NAME': 6,
               'B-DATE': 7, 'I-DATE': 8}  # Include all potential labels

def extract_features(tokens, max_seq_length):
    """Extracts features for a sequence of tokens."""
    features = []
    for i, token in enumerate(tokens):
        token_features = []

        # POS Tag
        doc = nlp(token)
        token_features.append(doc[0].pos_)

        # Character Features (Ensure everything is a string)
        token_features.append(str(token[0].isupper()))  # Is capitalized?
        token_features.append(str(token.isdigit()))  # Is numeric?

        features.append(token_features)

    # Pad features to ensure consistent sequence length (use '' for string padding)
    for i in range(len(features), max_seq_length):
        features.append(['', 'False', 'False'])  # Padding with empty strings and 'False' for boolean features

    return features

def pad_sequences(sequences, max_length, padding_value='O'):
    """Pads sequences to a specified length with the given padding value."""
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_sequences.append(seq + [padding_value] * (max_seq_length - len(seq)))
        else:
            padded_sequences.append(seq[:max_seq_length])  # Truncate if longer
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

# Load your annotated training and testing data
with open("../data/processed/annotated_data/train_data.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]
with open("../data/processed/annotated_data/test_data.jsonl", "r") as f:
    test_data = [json.loads(line) for line in f]

# Calculate max_seq_length (to be used in extract_features)
max_seq_length = max(len(sample["tokens"]) for sample in train_data)

# Extract features for training and testing data
X_train = [extract_features(sample["tokens"], max_seq_length) for sample in train_data]
X_test = [extract_features(sample["tokens"], max_seq_length) for sample in test_data]

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

# Convert labels to spaCy's format
train_examples = []
for i in range(len(X_train)):
    ents = []
    start = 0
    for j, label in enumerate(y_train_padded[i]):
        if label == "O":
            if start != -1:
                ents.append((start, j, y_train_padded[i][start]))
                start = -1
        elif label.startswith("B-"):
            if start != -1:
                ents.append((start, j, y_train_padded[i][start]))
            start = j
    if start != -1:
        ents.append((start, len(y_train_padded[i]), y_train_padded[i][start]))
    train_examples.append(Example.from_dict(nlp.make_doc(" ".join(train_data[i]["tokens"])), {"entities": ents}))

# ------------------MODEL TRAINING-----------------
# Get the existing NER pipe
ner = nlp.get_pipe("ner")

# Add new labels to the NER model
for label in label_encoder.classes_:
    ner.add_label(label)

# Disable other pipes and re-train the NER model
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    for _ in range(10):
        losses = {}
        for batch in spacy.util.minibatch(train_examples, size=32):
            nlp.update(batch, sgd=optimizer, losses=losses)
        print(losses)

# Evaluate the model
examples = []
for i in range(len(X_test)):
    ents = []
    start = 0
    for j, label in enumerate(y_test_padded[i]):
        if label == "O":
            if start != -1:
                ents.append((start, j, y_test_padded[i][start]))
                start = -1
        elif label.startswith("B-"):
            if start != -1:
                ents.append((start, j, y_test_padded[i][start]))
            start = j
    if start != -1:
        ents.append((start, len(y_test_padded[i]), y_test_padded[i][start]))
    examples.append(Example.from_dict(nlp.make_doc(" ".join(test_data[i]["tokens"])), {"entities": ents}))

scorer = Scorer()
scores = scorer.score(examples)
print(scores)

# Save the model
nlp.to_disk('../models/citation_extractor_spacy_ner')
