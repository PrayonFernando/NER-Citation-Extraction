import json
import joblib
import spacy
import pickle
from src.models.citation_model import preprocess_text
import fitz  # Import PyMuPDF
from sklearn.preprocessing import LabelEncoder
from src.preprocessing.annotation import annotate_data  # Import your annotation function

# Load the model
with open("../models/citation_classifier.pkl", "rb") as f:
    citation_classifier = pickle.load(f)
with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("../models/citation_extractor_crf.pkl", "rb") as f:
    crf_model = joblib.load(f)

# Load the label encoder from feature_extraction.py
with open("../models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")


# Extract Features Function (same as in feature_extraction.py)

def extract_features(tokens, citation_classifier, tfidf_vectorizer, max_seq_length):
    """Extracts features for a sequence of tokens and returns a list of lists of features."""
    features = []
    if tokens:
        for i, token in enumerate(tokens):
            token_features = []
            if token:
                # POS Tag
                doc = nlp(token)
                token_features.append(doc[0].pos_)

                # Character Features
                token_features.append(str(token[0].isupper()))
                token_features.append(str(token.isdigit()))

                # Get predictions from citation classifier only for non-padded tokens
                text_tfidf = tfidf_vectorizer.transform([token])
                prob = citation_classifier.predict_proba(text_tfidf)[0][1]
            else:
                # Set default values for empty tokens
                token_features.extend(['', 'False', 'False'])
                prob = 0.0

            token_features.append(str(prob))  # Add citation probability as a feature
            features.append(token_features)

    # Pad features to ensure consistent sequence length (use 0 for numeric padding)
    for i in range(len(features), max_seq_length):
        features.append(['', 0, 0, '0.0'])

    # Convert all features to strings
    features = [[str(item) for item in feature] for feature in features]
    return features


def predict_citations(tokens, max_seq_length):  # Removed 'text' from arguments
    """Predicts citations in a given list of tokens."""

    print(tokens)
    # Extract features
    X = extract_features(tokens, citation_classifier, tfidf_vectorizer, max_seq_length)

    # Predict with CRF model
    y_pred = crf_model.predict_single(X)

    # Invert label encoding to get original string labels
    predicted_labels = label_encoder.inverse_transform(y_pred)  # Ensure this is loaded correctly

    # Combine tokens and predicted labels
    predictions = list(zip(tokens, predicted_labels))
    print("Predictions:", predictions)

    # Post-process to get actual citations
    citations = []
    current_citation = []
    for token, label in predictions:
        if label == "B-CITATION":
            if current_citation:
                citations.append(" ".join(current_citation))
                current_citation = []
            current_citation.append(token)
        elif label == "I-CITATION":
            current_citation.append(token)
    if current_citation:  # Append any leftover citation at the end
        citations.append(" ".join(current_citation))

    return citations


def predict_citations_from_pdf(pdf_path, max_seq_length):
    """Predicts citations from a PDF file."""
    with open(pdf_path, "rb") as file:
        text = extract_text_from_pdf(file)

    # Annotate the text to get tokens and labels
    annotated_text = annotate_data([{'full_text': text}])

    # Assuming there is only 1 case in the data
    tokens = annotated_text[0]['tokens']

    # Removed excess arguments
    return predict_citations(tokens, max_seq_length)


def extract_text_from_pdf(file):
    """Extracts text from a PDF file using PyMuPDF."""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf_document:
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text


# Example usage (replace with your PDF file path):
pdf_file_path = "../data/raw/test_doc1.pdf"
max_seq_length = 100  # Adjust this based on your data
predicted_citations = predict_citations_from_pdf(pdf_file_path, max_seq_length)
print(predicted_citations)
