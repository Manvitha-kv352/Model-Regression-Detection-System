
import json
import os

from evaluation_runner import (
    config,
    dataset,
    create_fingerprint,
    save_results
)

old_path = "evaluations/v1_dataset_v2_checkpoint.json"
new_path = "evaluations/v1_dataset_v2_30_checkpoint.json"

if os.path.exists(new_path):
    raise FileExistsError("New checkpoint already exists.")

with open(old_path, "r", encoding="utf-8") as file:
    old_checkpoint = json.load(file)

# Reconstruct the original 28-case dataset
original_dataset = dict(dataset)
original_dataset["test_cases"] = [
    case for case in dataset["test_cases"]
    if case["id"] not in ("TC011", "TC012")
]

# Check that the old prompt and all original data are unchanged
original_fingerprint = create_fingerprint(
    config, original_dataset
)

if original_fingerprint != old_checkpoint["fingerprint"]:
    raise ValueError(
        "Original dataset or prompt has changed. "
        "Cannot safely reuse previous results."
    )

old_results = old_checkpoint["results"]

old_ids = {r["test_id"] for r in old_results}
expected_ids = {
    case["id"] for case in original_dataset["test_cases"]
}

if (
    len(dataset["test_cases"]) != 30
    or len(old_results) != 28
    or len(old_ids) != 28
    or old_ids != expected_ids
):
    raise ValueError("Dataset or checkpoint IDs do not match.")

new_fingerprint = create_fingerprint(config, dataset)

save_results(
    old_results,
    new_path,
    new_fingerprint
)

print("Fingerprint verification: PASSED")
print("Previous results reused:", len(old_results))
print("New cases remaining:", 30 - len(old_results))
print("New checkpoint:", new_path)
