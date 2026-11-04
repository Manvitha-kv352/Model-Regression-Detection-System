
import json

with open("datasets/golden_dataset_v1.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)

print("Total emails:", len(dataset))

for item in dataset:
    print(item["id"], 
    item["expected_category"],
    item["expected_summary"])
