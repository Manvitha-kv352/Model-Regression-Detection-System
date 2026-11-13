
import asyncio
import hashlib
import json
import os

from dotenv import load_dotenv
from groq import AsyncGroq, RateLimitError

from classifier import classify_email
from config import load_prompt_config


load_dotenv()

# Load prompt configuration
config = load_prompt_config("prompts/v1.yaml")

# Load golden dataset
with open(
    "datasets/golden_dataset_v3.json",
    "r",
    encoding="utf-8"
) as file:
    dataset = json.load(file)


# Generate fingerprint for prompt configuration and dataset
def create_fingerprint(config, dataset):
    content = {
        "prompt_config": {
            "version": config.version,
            "model": config.model,
            "temperature": config.temperature,
            "system_prompt": config.system_prompt
        },
        "dataset": dataset
    }

    serialized = json.dumps(
        content,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# Save evaluation results to JSON
def save_results(results, file_path, fingerprint):
    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    data = {
        "fingerprint": fingerprint,
        "results": results
    }

    temporary_path = file_path + ".tmp"

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    os.replace(temporary_path, file_path)


# Load saved results and validate fingerprint
def load_results(file_path, expected_fingerprint):
    if not os.path.exists(file_path):
        return []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Saved results use the old format. "
            "Use a new results filename."
        )

    if data.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Saved results belong to a different prompt or dataset. "
            "Use a new results filename."
        )

    results = data.get("results")

    if not isinstance(results, list):
        raise ValueError(
            "Invalid checkpoint: results must be a list."
        )

    return results


# Evaluate one test case
async def evaluate_case(test_case, config, client):
    result = await classify_email(
        test_case["input"],
        config,
        client
    )

    parsed = result["parsed_output"]

    predicted_category = (
        parsed.get("category")
        if isinstance(parsed, dict)
        else None
    )

    expected_category = test_case["expected_output"]["category"]

    # Dimension 1: Exact category match (binary)
    category_match = int(
        result["error"] is None
        and predicted_category == expected_category
    )

    status = (
        "PASS"
        if category_match == 1
        else "FAIL"
    )

    return {
        "test_id": test_case["id"],
        "expected_category": expected_category,
        "raw_output": result["raw_output"],
        "predicted_category": predicted_category,
        "predicted_summary": (
            parsed.get("summary")
            if isinstance(parsed, dict)
            else None
        ),
        "error": result["error"],
        "category_match": category_match,
        "latency_ms": result["latency_ms"],
        "token_usage": result["token_usage"],
        "status": status
    }


# Run evaluation with batching, checkpointing, and retries
async def run_evaluation(config, dataset):
    test_cases = dataset["test_cases"]

    selected_cases = test_cases[:3]

    batch_size = 3
    batch_delay = 20

    # Keep the existing completed 30-case checkpoint
    output_path = (
        "evaluations/v1_dataset_v3_30_checkpoint.json"
    )
    fingerprint = create_fingerprint(config, dataset)

    # Load previously saved results
    saved_results = load_results(
        output_path,
        fingerprint
    )

    # Keep results indexed by test ID
    results_by_id = {
        result["test_id"]: result
        for result in saved_results
    }

    # PASS and FAIL are completed evaluations
    completed_ids = {
        result["test_id"]
        for result in saved_results
        if result["status"] in ("PASS", "FAIL")
    }

    # Select only cases that still need evaluation
    pending_cases = [
        test_case
        for test_case in selected_cases
        if test_case["id"] not in completed_ids
    ]

    print("Selected test cases:", len(selected_cases))
    print(
        "Already completed:",
        len(selected_cases) - len(pending_cases)
    )
    print(
        "Remaining to evaluate:",
        len(pending_cases)
    )

    # Avoid API calls when everything is already completed
    if not pending_cases:
        print(
            "All selected test cases are already completed."
        )

        return [
            results_by_id[test_case["id"]]
            for test_case in selected_cases
        ]

    semaphore = asyncio.Semaphore(3)

    async with AsyncGroq(
        api_key=os.getenv("GROQ_API_KEY")
    ) as client:

        async def limited_evaluate(test_case):
            max_retries = 5

            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        return await evaluate_case(
                            test_case,
                            config,
                            client
                        )

                except RateLimitError as exc:
                    if attempt == max_retries - 1:
                        return {
                            "test_id": test_case["id"],
                            "expected_category": (
                                test_case["expected_output"]["category"]
                            ),
                            "raw_output": None,
                            "predicted_category": None,
                            "predicted_summary": None,
                            "error": (
                                "Rate limit exceeded after retries: "
                                f"{exc}"
                            ),
                            "category_match": 0,
                            "latency_ms": None,
                            "token_usage": {
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "total_tokens": None
                            },
                            "status": "ERROR"
                        }

                    wait_time = min(
                        3 * (2 ** attempt),
                        30
                    )

                    print(
                        f"Rate limit for {test_case['id']}. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    await asyncio.sleep(wait_time)

        # Process pending cases in batches
        for i in range(
            0,
            len(pending_cases),
            batch_size
        ):
            batch = pending_cases[
                i:i + batch_size
            ]

            print(
                f"Processing batch "
                f"{i // batch_size + 1}: "
                f"{len(batch)} test cases"
            )

            tasks = [
                limited_evaluate(test_case)
                for test_case in batch
            ]

            batch_results = await asyncio.gather(
                *tasks
            )

            # Add or update results without duplicate IDs
            for result in batch_results:
                results_by_id[
                    result["test_id"]
                ] = result

            # Preserve dataset order
            results = [
                results_by_id[test_case["id"]]
                for test_case in selected_cases
                if test_case["id"] in results_by_id
            ]

            # Save after each batch
            save_results(
                results,
                output_path,
                fingerprint
            )

            print(
                f"Saved {len(results)} results "
                f"to {output_path}"
            )

            # Delay only when another batch remains
            if i + batch_size < len(pending_cases):
                print(
                    f"Waiting {batch_delay} seconds "
                    "before next batch..."
                )

                await asyncio.sleep(batch_delay)

    # Return completed results in dataset order
    return [
        results_by_id[test_case["id"]]
        for test_case in selected_cases
        if test_case["id"] in results_by_id
    ]


# Main function
async def main():
    print("Prompt version:", config.version)
    print(
        "Dataset version:",
        dataset["dataset_version"]
    )
    print(
        "Total test cases:",
        len(dataset["test_cases"])
    )

    results = await run_evaluation(
        config,
        dataset
    )

    correct_predictions = sum(
        1
        for result in results
        if result.get("category_match") == 1
        or (
            "category_match" not in result
            and result["status"] == "PASS"
        )
    )

    total_predictions = len(results)

    accuracy = (
        correct_predictions / total_predictions * 100
        if total_predictions > 0
        else 0
    )

    print("\n--- Evaluation Summary ---")
    print(
        "Correct predictions:",
        correct_predictions
    )
    print(
        "Total evaluated:",
        total_predictions
    )
    print(
        f"Classification accuracy: "
        f"{accuracy:.2f}%"
    )

    print("\n--- Individual Test Results ---")

    for result in results:
        print("Test ID:", result["test_id"])
        print(
            "Expected category:",
            result["expected_category"]
        )
        print(
            "Predicted category:",
            result["predicted_category"]
        )
        print(
            "Predicted summary:",
            result["predicted_summary"]
        )
        print(
            "Raw output:",
            result["raw_output"]
        )
        print(
            "Category match:",
            result.get(
                "category_match",
                int(result["status"] == "PASS")
            )
        )
        print(
            "Latency (ms):",
            result.get("latency_ms")
        )
        print(
            "Token usage:",
            result.get("token_usage")
        )
        print(
            "Error:",
            result["error"]
        )
        print(
            "Status:",
            result["status"]
        )
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
