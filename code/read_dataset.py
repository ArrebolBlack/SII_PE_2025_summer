import json
import jsonlines
import csv
from typing import List, Dict, Any

# File paths
val_path = "E:\\PE_Exam\\val.jsonl"
json_output_path = "E:\\PE_Exam\\val.json"
csv_output_path = "E:\\PE_Exam\\val.csv"


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    with jsonlines.open(filepath) as reader:
        return list(reader)


def save_as_json(data: List[Dict[str, Any]], filepath: str):
    """Save data as a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_as_csv(data: List[Dict[str, Any]], filepath: str):
    """Save data as a CSV file."""
    # Determine headers from the first entry
    if not data:
        print("No data to write to CSV.")
        return

    # Extract keys from the first dictionary to use as headers
    headers = list(data[0].keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


# Load the JSONL dataset
val_dataset = load_jsonl(val_path)

# Save as JSON
save_as_json(val_dataset, json_output_path)
print(f"✅ JSON file has been generated: {json_output_path}")

# Save as CSV
save_as_csv(val_dataset, csv_output_path)
print(f"✅ CSV file has been generated: {csv_output_path}")
