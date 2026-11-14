
import json
from pathlib import Path

SOURCE = Path("datasets/golden_dataset_v1.json")
DESTINATION = Path("datasets/golden_dataset_v1_normalized.json")

with SOURCE.open("r", encoding="utf-8") as file:
    original = json.load(file)

if not isinstance(original, list):
    raise ValueError("Expected Dataset v1 to be a list.")

normalized_cases = []

for case in original:
    normalized_cases.append({
        **case,
        "id": f"TC{int(case['id']):03d}",
        "input": case["email"]
    })

normalized_dataset = {
    "dataset_version": "v1",
    "test_cases": normalized_cases
}

with DESTINATION.open("w", encoding="utf-8") as file:
    json.dump(
        normalized_dataset,
        file,
        indent=2,
        ensure_ascii=False
    )

print("Dataset v1 normalized successfully!")
print("Total cases:", len(normalized_cases))
print("Saved to:", DESTINATION)
