import json
import os
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# def annotate_data(data):
#     """Annotates legal case data with BIO tags."""
#     annotated_data = []
#
#     for case in data:
#         # Process the 'name' field
#         name_tokens = case["name"].split()
#         name_labels = ["B-CASE_NAME"] + ["I-CASE_NAME"] * (len(name_tokens) - 1)
#
#         # Process the 'court_name' field
#         court_tokens = case["court"]["name"].split()
#         court_labels = ["B-COURT_NAME"] + ["I-COURT_NAME"] * (len(court_tokens) - 1)
#
#         # Process the 'decision_date' field
#         date_tokens = case["decision_date"].split("-")
#         date_labels = ["B-DATE"] + ["I-DATE"] * (len(date_tokens) - 1)
#
#         # Process the 'citations' field
#         citation_tokens = []
#         citation_labels = []
#         for citation in case.get("citations", []):
#             cite_text = citation["cite"]
#             tokens = cite_text.split()
#             labels = ["B-CITATION"] + ["I-CITATION"] * (len(tokens) - 1)
#             citation_tokens.extend(tokens)
#             citation_labels.extend(labels)
#
#         # Combine tokens and labels from all fields
#         all_tokens = name_tokens + court_tokens + date_tokens + citation_tokens
#         all_labels = name_labels + court_labels + date_labels + citation_labels
#
#         annotated_data.append({
#             "tokens": all_tokens,
#             "labels": all_labels
#         })
#
#     return annotated_data


def annotate_data(data):
    """Annotates citations in legal case data with BIO tagging."""
    annotated_data = []

    for case in data:
        text = case["name"]
        tokens = text.split()
        labels = ["O"] * len(tokens)

        for citation in case["citations"]:
            cite_text = citation["cite"]
            # Use regex to find all occurrences of the citation in the text
            for match in re.finditer(re.escape(cite_text), text):
                start, end = match.span()
                cite_tokens = text[start:end].split()
                for i, token in enumerate(cite_tokens):
                    if i == 0:
                        labels[tokens.index(token)] = "B-CITATION"
                    else:
                        labels[tokens.index(token)] = "I-CITATION"

        annotated_data.append({"tokens": tokens, "labels": labels})

    return annotated_data

# Specify the directory to save the annotated data
save_directory = "../../data/processed/annotated_data"  # Create this directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Load your JSON data
with open("../../data/raw/CasesMetadata.json", "r") as f:
    data = json.load(f)

# Annotate the data
annotated_data = annotate_data(data)

# Create and fit LabelEncoder on all the labels
all_labels = [label for sample in annotated_data for label in sample["labels"]]
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Save the updated label encoder
with open('../models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Split the data into training and testing sets (80/20 split)
train_data, test_data = train_test_split(annotated_data, test_size=0.2, random_state=42)

# Save the training data
train_file = os.path.join(save_directory, "train_data.jsonl")
with open(train_file, "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

print(f"Training data saved to {train_file}")

# Save the testing data
test_file = os.path.join(save_directory, "test_data.jsonl")
with open(test_file, "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

print(f"Testing data saved to {test_file}")

