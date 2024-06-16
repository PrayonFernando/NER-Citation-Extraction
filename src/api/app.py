# src/api/app.py
# from ..src.extraction.rule_based import extract_case_citations  # Absolute import
import pickle

from src.preprocessing.preprocess import preprocess_text

def mock_extract_endpoint(text):
    # Load the trained model
    with open('models/citation_classifier.pkl', 'rb') as f:
        model = pickle.load(f)

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
