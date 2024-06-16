import pandas as pd
import re
import os


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


def create_dataset(df, text_col, label_col):
    """
    Creates a dataset for training or testing from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        text_col (str): The name of the column containing the text.
        label_col (str): The name of the column containing the labels.

    Returns:
        list, list: A tuple containing two lists:
            - texts: The preprocessed texts.
            - labels: The corresponding labels.
    """
    texts = [preprocess_text(text) for text in df[text_col]]
    labels = df[label_col].tolist()
    return texts, labels
