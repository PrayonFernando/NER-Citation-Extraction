import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def annotate_data(data):
    """Annotates all fields in legal case data with BIO tags."""
    annotated_data = []

    for case in data:
        # Initialize an empty list to store labeled tokens for the entire case
        case_tokens = []
        case_labels = []

        # Process each field separately and store its tokens and labels
        for field_name, field_value in case.items():
            if isinstance(field_value,
                          list) and field_name != 'cites_to':  # Handle lists (except cites_to, as they are not necessary for our task)
                for item in field_value:
                    for sub_field_name, sub_field_value in item.items():
                        if sub_field_name == 'cite':  # Label citations with B-CITATION and I-CITATION
                            tokens = sub_field_value.split()
                            labels = ["B-CITATION"] + ["I-CITATION"] * (len(tokens) - 1)
                        else:  # Label all other tokens as O
                            tokens = str(sub_field_value).split()
                            labels = ["O"] * len(tokens)
                        case_tokens.extend(tokens)
                        case_labels.extend(labels)
            else:  # Label all other fields as O
                tokens = str(field_value).split()
                case_tokens.extend(tokens)
                case_labels.extend(["O"] * len(tokens))

        # Check for citations and label the entire case accordingly
        has_citation = any(label != 'O' for label in case_labels)
        annotated_data.append({
            "tokens": case_tokens,
            "labels": case_labels,
            "has_citation": has_citation
        })

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

