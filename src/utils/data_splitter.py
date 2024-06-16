import json
from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(json_file_path, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    # Flatten the citations list for each case
    flattened_citations = []
    for _, row in df.iterrows():
        for citation in row['citations']:
            flattened_citations.append({
                'Case Name': row['name'],
                'type': citation['type'],
                'cite': citation['cite']
            })

    # Create a DataFrame from the flattened citations
    df = pd.DataFrame(flattened_citations)

    # Extract citation text from the dictionaries in the Citation column
    df['Citation'] = df['cite']

    # Create a new column called `label`
    df['label'] = (df['type'] == 'official') | (df['type'] == 'parallel').astype(int)

    # Create a new column called `is_parallel`
    df['is_parallel'] = (df['type'] == 'parallel').astype(bool)

    # Drop unused columns
    df = df[['Case Name', 'Citation', 'label', 'is_parallel']]

    train_data, test_data = train_test_split(df, test_size=test_size, stratify=df['is_parallel'],
                                             random_state=random_state)
    return train_data, test_data

if __name__ == "__main__":
    json_file_path = '../../data/raw/CasesMetadata.json'
    train_data, test_data = split_data(json_file_path)

    train_data.to_csv('../../data/processed/train.csv', index=False)
    test_data.to_csv('../../data/processed/test.csv', index=False)


# Read and display the first 5 rows of train.csv and test.csv
train_df = pd.read_csv('../../data/processed/train.csv')
test_df = pd.read_csv('../../data/processed/test.csv')

print("Training Set:")
print(train_df.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\nTesting Set:")
print(test_df.head().to_markdown(index=False, numalign="left", stralign="left"))
