
import json
from pathlib import Path

from scoring import score_case, summarize_scores
from evaluation_runner import create_fingerprint, config


# --------------------------------------------------
# File paths
# --------------------------------------------------

DATASET_V2_PATH = Path(
    "datasets/golden_dataset_v2.json"
)

DATASET_V3_PATH = Path(
    "datasets/golden_dataset_v3.json"
)

CLASSIFIER_PATH = Path(
    "evaluations/v1_dataset_v2_30_checkpoint.json"
)

JUDGED_PATH = Path(
    "evaluations/v1_dataset_v2_30_judged.json"
)

OUTPUT_PATH = Path(
    "evaluations/v1_dataset_v3_scored_report.json"
)


# --------------------------------------------------
# Load JSON
# --------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# Validate source checkpoints and datasets
# --------------------------------------------------

def validate_sources(
    v2_dataset,
    v3_dataset,
    classifier_data,
    judged_data,
):
    expected_fingerprint = create_fingerprint(
        config,
        v2_dataset
    )

    if classifier_data.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Classifier checkpoint fingerprint does not "
            "match Prompt v1 and Dataset v2."
        )

    if judged_data.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Judge checkpoint fingerprint does not "
            "match Prompt v1 and Dataset v2."
        )

    v2_cases = {
        case["id"]: case
        for case in v2_dataset["test_cases"]
    }

    v3_cases = {
        case["id"]: case
        for case in v3_dataset["test_cases"]
    }

    if len(v2_cases) != len(v2_dataset["test_cases"]):
        raise ValueError(
            "Duplicate test IDs found in Dataset v2."
        )

    if len(v3_cases) != len(v3_dataset["test_cases"]):
        raise ValueError(
            "Duplicate test IDs found in Dataset v3."
        )

    if set(v2_cases) != set(v3_cases):
        raise ValueError(
            "Dataset v2 and v3 contain different test IDs. "
            "Cached predictions cannot be safely reused."
        )

    for test_id, new_case in v3_cases.items():
        old_case = v2_cases[test_id]

        if new_case["input"] != old_case["input"]:
            raise ValueError(
                f"Input changed for {test_id}. "
                "Cached prediction cannot be reused."
            )

    if v3_dataset.get("dataset_version") != "v3":
        raise ValueError(
            "Expected dataset_version 'v3'."
        )

    if v3_dataset.get("routing_policy_version") != "v1":
        raise ValueError(
            "Expected routing_policy_version 'v1'."
        )

    print("Source validation: PASSED")
    print("Checkpoint fingerprints: VERIFIED")
    print("Dataset input consistency: VERIFIED")


# --------------------------------------------------
# Build result lookup
# --------------------------------------------------

def build_result_lookup(data, source_name):
    results = data.get("results")

    if not isinstance(results, list):
        raise ValueError(
            f"{source_name} must contain a results list."
        )

    lookup = {}

    for result in results:
        test_id = result.get("test_id")

        if not test_id:
            raise ValueError(
                f"{source_name} contains a result "
                "without a test_id."
            )

        if test_id in lookup:
            raise ValueError(
                f"Duplicate result {test_id} "
                f"in {source_name}."
            )

        lookup[test_id] = result

    return lookup


# --------------------------------------------------
# Generate evaluation report
# --------------------------------------------------

def main():
    print("\nLoading evaluation data...")

    v2_dataset = load_json(DATASET_V2_PATH)
    v3_dataset = load_json(DATASET_V3_PATH)

    classifier_data = load_json(CLASSIFIER_PATH)
    judged_data = load_json(JUDGED_PATH)

    # Validate before scoring
    validate_sources(
        v2_dataset,
        v3_dataset,
        classifier_data,
        judged_data,
    )

    test_cases = v3_dataset["test_cases"]

    if len(test_cases) != 30:
        raise ValueError(
            "Expected 30 test cases in Dataset v3."
        )

    classifier_results = build_result_lookup(
        classifier_data,
        "Classifier checkpoint"
    )

    judged_results = build_result_lookup(
        judged_data,
        "Judge checkpoint"
    )

    expected_ids = {
        case["id"] for case in test_cases
    }

    if set(classifier_results) != expected_ids:
        raise ValueError(
            "Classifier results do not match "
            "Dataset v3 test IDs."
        )

    if set(judged_results) != expected_ids:
        raise ValueError(
            "Judge results do not match "
            "Dataset v3 test IDs."
        )

    scored_results = []

    for case in test_cases:
        test_id = case["id"]

        classifier_result = classifier_results[test_id]
        judged_result = judged_results[test_id]

        judgment = {
            "summary_relevance": judged_result.get(
                "summary_relevance"
            ),
            "judge_reason": judged_result.get(
                "judge_reason"
            ),
            "judge_error": judged_result.get(
                "judge_error"
            ),
        }

        scored = score_case(
            test_case=case,
            classifier_result=classifier_result,
            judgment=judgment,
        )

        scored_results.append(scored)

    # Overall metrics
    summary = summarize_scores(scored_results)

    # Accuracy grouped by expected category
    category_stats = {}

    for result in scored_results:
        category = result["expected_category"]

        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "correct": 0,
            }

        category_stats[category]["total"] += 1
        category_stats[category]["correct"] += (
            result["scores"]["category_match"]
        )

    summary["accuracy_by_category"] = {
        category: {
            "total": values["total"],
            "correct": values["correct"],
            "accuracy_percent": round(
                values["correct"] / values["total"] * 100,
                2
            ),
        }
        for category, values in category_stats.items()
    }

    # Identify failed category predictions
    failures = [
        result
        for result in scored_results
        if result["scores"]["category_match"] == 0
    ]

    # Assemble final report
    report = {
        "prompt_version": config.version,
        "dataset_version": v3_dataset["dataset_version"],
        "routing_policy_version": v3_dataset.get(
            "routing_policy_version"
        ),
        "evaluation_type": "cached_predictions_rescored",
        "source_dataset_version": "v2",
        "source_fingerprint": classifier_data["fingerprint"],
        "summary": summary,
        "results": scored_results,
    }

    # Save report without modifying source checkpoints
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    temporary_path.replace(OUTPUT_PATH)

    # --------------------------------------------------
    # Terminal report
    # --------------------------------------------------

    print("\n--- Dataset v3 Evaluation Report ---")
    print("Prompt version:", report["prompt_version"])
    print("Dataset version:", report["dataset_version"])
    print(
        "Routing policy:",
        report["routing_policy_version"]
    )
    print("Total cases:", summary["total_cases"])

    print(
        "Category accuracy:",
        f'{summary["category_accuracy"]:.2f}%'
    )

    print(
        "Average summary relevance:",
        summary["average_summary_relevance"]
    )

    print("\n--- Accuracy by Difficulty ---")

    for difficulty, accuracy in summary[
        "difficulty_accuracy"
    ].items():
        print(
            f"{difficulty}: {accuracy:.2f}%"
        )

    print("\n--- Accuracy by Category ---")

    for category, values in summary[
        "accuracy_by_category"
    ].items():
        print(
            f'{category}: '
            f'{values["correct"]}/{values["total"]} '
            f'({values["accuracy_percent"]:.2f}%)'
        )

    print("\n--- Failed Cases ---")

    if not failures:
        print("No category mismatches.")
    else:
        for result in failures:
            print(
                result["test_id"],
                "| Expected:",
                result["expected_category"],
                "| Predicted:",
                result["predicted_category"],
                "| Summary relevance:",
                result["scores"]["summary_relevance"],
            )

    print("\nReport saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
