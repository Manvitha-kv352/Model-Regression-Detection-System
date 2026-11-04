
import json

with open("datasets/golden_dataset_v2.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)

print("Dataset version:", dataset["dataset_version"])
print("Total cases:", len(dataset["test_cases"]))

for case in dataset["test_cases"]:
    print(
        case["id"],
        case["expected_output"]["category"],
        case["difficulty"]
    )
