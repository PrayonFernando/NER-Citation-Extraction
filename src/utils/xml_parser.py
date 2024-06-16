import os
import json
import xml.etree.ElementTree as ET

def extract_citations_from_xml(xml_file_path, json_data):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    ns = {'case': 'http://nrs.harvard.edu/urn-3:HLS.Libr.US_Case_Law.Schema.Case:v1'}

    case_data = {
        'id': None,  # Add id field to match the JSON structure
        'name': root.find('.//case:name', ns).text,
        'name_abbreviation': None,  # Placeholder as this is not directly in the XML
        'decision_date': root.find('.//case:decisiondate', ns).text,
        'docket_number': root.find('.//case:docketnumber', ns).text,  # Assuming docketnumber exists
        'first_page': None,  # These are not directly in the XML, you might need to infer them
        'last_page': None,
        'citations': [],
        'court': {'name_abbreviation': None, 'id': None, 'name': None},
        'jurisdiction': {'id': None, 'name_long': None, 'name': None},
        'cites_to': []  # This will be filled later
    }

    for citation in root.findall('.//case:citation', ns):
        case_data['citations'].append({
            'type': citation.attrib.get('category', 'unknown'),  # Default to 'unknown' if category not present
            'cite': citation.text
        })

    json_data.append(case_data)  # Add the extracted data to the list

def parse_xml_files(xml_directory, json_filename):
    json_data = []

    for filename in os.listdir(xml_directory):
        if filename.endswith(".xml"):
            filepath = os.path.join(xml_directory, filename)
            extract_citations_from_xml(filepath, json_data)

    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=4)

# Main execution (replace paths with your actual values)
if __name__ == "__main__":
    xml_directory = "../../data/raw/alaska-fed/"
    json_filename = "../../data/processed/extracted_citations.json"
    parse_xml_files(xml_directory, json_filename)
