
import json
from pathlib import Path


DATASET_PATH = Path("datasets/golden_dataset_v3.json")


def load_test_case(test_id):
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    for case in dataset["test_cases"]:
        if case["id"] == test_id:
            return case

    raise ValueError(f"Test case {test_id} not found")


def evaluate_category(expected_category, predicted_category):
    return int(expected_category == predicted_category)


def validate_schema(output):
    if not isinstance(output, dict):
        return False

    if set(output.keys()) != {"category", "summary"}:
        return False

    if output["category"] not in {
        "billing", "technical", "account", "general"
    }:
        return False

    if not isinstance(output["summary"], str):
        return False

    if not output["summary"].strip():
        return False

    return True


def run_tests():
    case = load_test_case("TC002")

    expected_category = case["expected_output"]["category"]

    print("Test case:", case["id"])
    print("Customer email:", case["input"])
    print("Expected category:", expected_category)

    fixtures = [
        {
            "name": "Correct output",
            "output": {
                "category": "billing",
                "summary": (
                    "Customer wants to cancel their subscription "
                    "because the application keeps crashing."
                )
            },
            "expected_schema_valid": True,
            "expected_category_match": 1
        },
        {
            "name": "Wrong primary category",
            "output": {
                "category": "technical",
                "summary": (
                    "Customer wants to cancel their subscription "
                    "because the application keeps crashing."
                )
            },
            "expected_schema_valid": True,
            "expected_category_match": 0
        },
        {
            "name": "Incorrect summary",
            "output": {
                "category": "billing",
                "summary": "Customer wants to upgrade to premium."
            },
            "expected_schema_valid": True,
            "expected_category_match": 1
        },
        {
            "name": "Invalid category",
            "output": {
                "category": "subscription",
                "summary": "Customer requests cancellation."
            },
            "expected_schema_valid": False,
            "expected_category_match": 0
        },
        {
            "name": "Missing summary",
            "output": {
                "category": "billing"
            },
            "expected_schema_valid": False,
            "expected_category_match": 1
        }
    ]

    passed = 0

    for fixture in fixtures:
        output = fixture["output"]

        schema_valid = validate_schema(output)

        category_match = evaluate_category(
            expected_category,
            output.get("category")
        )

        test_passed = (
            schema_valid == fixture["expected_schema_valid"]
            and category_match == fixture["expected_category_match"]
        )

        if test_passed:
            passed += 1

        print("\n---", fixture["name"], "---")
        print("Schema valid:", schema_valid)
        print("Category match:", category_match)
        print("Expected behavior detected:", test_passed)

    print("\n==========================")
    print(f"Validation tests passed: {passed}/{len(fixtures)}")
    print("==========================")


if __name__ == "__main__":
    run_tests()
