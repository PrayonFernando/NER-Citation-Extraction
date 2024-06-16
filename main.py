import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (two levels up)
project_root = os.path.dirname(os.path.dirname(current_dir))

# Add the project root to the Python path
sys.path.insert(0, project_root)


from src.api.app import mock_extract_endpoint

if __name__ == '__main__':
    text = input("Enter legal text: ")
    result = mock_extract_endpoint(text)

    print("Extracted Citations:")
    for citation in result['citations']:
        print(citation)
