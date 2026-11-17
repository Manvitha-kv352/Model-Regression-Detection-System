
import asyncio
import hashlib
import json
import os

from dotenv import load_dotenv
from groq import AsyncGroq, RateLimitError

from classifier import classify_email
from config import load_prompt_config


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

PROMPT_CONFIG_PATH = os.getenv(
    "PROMPT_CONFIG_PATH",
    "prompts/v2.yaml"
)

config = load_prompt_config(PROMPT_CONFIG_PATH)

DATASET_PATH = "datasets/golden_dataset_v3.json"

OUTPUT_PATH = os.getenv(
    "EVALUATION_CHECKPOINT_PATH",
    "evaluations/v2_dataset_v3_30_checkpoint.json"
)

BATCH_SIZE = 3
BATCH_DELAY = 20
MAX_RETRIES = 5


# --------------------------------------------------
# Load and validate dataset
# --------------------------------------------------

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)

if not isinstance(dataset, dict):
    raise ValueError("Dataset must be a JSON object.")

if not isinstance(dataset.get("test_cases"), list):
    raise ValueError("Dataset must contain a test_cases list.")

test_ids = []

for case in dataset["test_cases"]:
    if not isinstance(case, dict):
        raise ValueError("Each test case must be an object.")

    required_fields = ("id", "input", "expected_output")

    if not all(field in case for field in required_fields):
        raise ValueError(
            f"Dataset case missing required fields: {case!r}"
        )

    if not isinstance(case["expected_output"], dict):
        raise ValueError(
            f"Invalid expected_output for {case['id']}"
        )

    if "category" not in case["expected_output"]:
        raise ValueError(
            f"Missing expected category for {case['id']}"
        )

    test_ids.append(case["id"])

if len(test_ids) != len(set(test_ids)):
    raise ValueError("Dataset contains duplicate test IDs.")


# --------------------------------------------------
# Fingerprint generation
# --------------------------------------------------

def create_fingerprint(config, dataset):
    content = {
        "prompt_config": {
            "version": config.version,
            "model": config.model,
            "temperature": config.temperature,
            "system_prompt": config.system_prompt,
        },
        "dataset": dataset,
    }

    serialized = json.dumps(
        content,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# Atomic checkpoint saving
# --------------------------------------------------

def save_results(results, file_path, fingerprint):
    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    data = {
        "fingerprint": fingerprint,
        "results": results,
    }

    temporary_path = file_path + ".tmp"

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    os.replace(temporary_path, file_path)


# --------------------------------------------------
# Load and validate checkpoint
# --------------------------------------------------

def load_results(file_path, expected_fingerprint):
    if not os.path.exists(file_path):
        return []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Saved checkpoint uses an unsupported format. "
            "Use a new checkpoint filename."
        )

    if data.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Checkpoint belongs to a different prompt "
            "or dataset. Use a new checkpoint filename."
        )

    results = data.get("results")

    if not isinstance(results, list):
        raise ValueError(
            "Invalid checkpoint: results must be a list."
        )

    seen_ids = set()

    for result in results:
        if not isinstance(result, dict):
            raise ValueError(
                "Invalid checkpoint result entry."
            )

        test_id = result.get("test_id")

        if test_id is None:
            raise ValueError(
                "Checkpoint result missing test_id."
            )

        if test_id in seen_ids:
            raise ValueError(
                f"Duplicate checkpoint test ID: {test_id}"
            )

        seen_ids.add(test_id)

    return results


# --------------------------------------------------
# Evaluate a single test case
# --------------------------------------------------

async def evaluate_case(test_case, config, client):
    result = await classify_email(
        test_case["input"],
        config,
        client,
    )

    parsed = result.get("parsed_output")

    predicted_category = (
        parsed.get("category")
        if isinstance(parsed, dict)
        else None
    )

    predicted_summary = (
        parsed.get("summary")
        if isinstance(parsed, dict)
        else None
    )

    expected_category = (
        test_case["expected_output"]["category"]
    )

    error = result.get("error")

    category_match = int(
        error is None
        and predicted_category == expected_category
    )

    # API or parsing errors are marked ERROR.
    # Genuine incorrect classifications are FAIL.
    if error is not None:
        status = "ERROR"
    elif category_match == 1:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "test_id": test_case["id"],
        "expected_category": expected_category,
        "raw_output": result.get("raw_output"),
        "predicted_category": predicted_category,
        "predicted_summary": predicted_summary,
        "error": error,
        "category_match": category_match,
        "latency_ms": result.get("latency_ms"),
        "token_usage": result.get("token_usage"),
        "status": status,
    }


# --------------------------------------------------
# Error result helper
# --------------------------------------------------

def create_error_result(test_case, message):
    return {
        "test_id": test_case["id"],
        "expected_category": (
            test_case["expected_output"]["category"]
        ),
        "raw_output": None,
        "predicted_category": None,
        "predicted_summary": None,
        "error": str(message),
        "category_match": 0,
        "latency_ms": None,
        "token_usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "status": "ERROR",
    }


# --------------------------------------------------
# Evaluation runner
# --------------------------------------------------

async def run_evaluation(config, dataset):
    test_cases = dataset["test_cases"]
    selected_cases = test_cases

    fingerprint = create_fingerprint(
        config,
        dataset,
    )

    saved_results = load_results(
        OUTPUT_PATH,
        fingerprint,
    )

    valid_ids = {
        case["id"] for case in selected_cases
    }

    results_by_id = {
        result["test_id"]: result
        for result in saved_results
        if result["test_id"] in valid_ids
    }

    # A genuine incorrect prediction is still a
    # completed evaluation. Errors are retryable.
    completed_ids = {
        result["test_id"]
        for result in saved_results
        if (
            result["test_id"] in valid_ids
            and result.get("status") in ("PASS", "FAIL")
            and result.get("error") is None
        )
    }

    pending_cases = [
        case
        for case in selected_cases
        if case["id"] not in completed_ids
    ]

    print("Prompt version:", config.version)
    print("Selected test cases:", len(selected_cases))
    print("Already completed:", len(completed_ids))
    print("Remaining to evaluate:", len(pending_cases))

    if not pending_cases:
        print(
            "All selected test cases are already completed."
        )

        return [
            results_by_id[case["id"]]
            for case in selected_cases
        ]

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Configure it in your environment."
        )

    semaphore = asyncio.Semaphore(BATCH_SIZE)

    async with AsyncGroq(api_key=api_key) as client:

        async def limited_evaluate(test_case):
            for attempt in range(MAX_RETRIES):
                try:
                    async with semaphore:
                        return await evaluate_case(
                            test_case,
                            config,
                            client,
                        )

                except RateLimitError as exc:
                    if attempt == MAX_RETRIES - 1:
                        return create_error_result(
                            test_case,
                            "Rate limit exceeded after "
                            f"{MAX_RETRIES} attempts: {exc}",
                        )

                    wait_time = min(
                        3 * (2 ** attempt),
                        30,
                    )

                    print(
                        f"Rate limit for {test_case['id']}. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    await asyncio.sleep(wait_time)

                except Exception as exc:
                    return create_error_result(
                        test_case,
                        f"{type(exc).__name__}: {exc}",
                    )

            return create_error_result(
                test_case,
                "Evaluation exhausted all retry attempts.",
            )

        for i in range(
            0,
            len(pending_cases),
            BATCH_SIZE,
        ):
            batch = pending_cases[
                i:i + BATCH_SIZE
            ]

            print(
                f"Processing batch "
                f"{i // BATCH_SIZE + 1}: "
                f"{len(batch)} test cases"
            )

            tasks = [
                limited_evaluate(case)
                for case in batch
            ]

            batch_results = await asyncio.gather(
                *tasks
            )

            for result in batch_results:
                results_by_id[
                    result["test_id"]
                ] = result

            # Preserve the original dataset order.
            results = [
                results_by_id[case["id"]]
                for case in selected_cases
                if case["id"] in results_by_id
            ]

            save_results(
                results,
                OUTPUT_PATH,
                fingerprint,
            )

            print(
                f"Saved {len(results)} results "
                f"to {OUTPUT_PATH}"
            )

            if i + BATCH_SIZE < len(pending_cases):
                print(
                    f"Waiting {BATCH_DELAY} seconds "
                    "before next batch..."
                )

                await asyncio.sleep(BATCH_DELAY)

    return [
        results_by_id[case["id"]]
        for case in selected_cases
        if case["id"] in results_by_id
    ]


# --------------------------------------------------
# Main
# --------------------------------------------------

async def main():
    print("\n--- LLM Regression Evaluation ---")

    print("Prompt version:", config.version)

    print(
        "Dataset version:",
        dataset.get("dataset_version", "unknown"),
    )

    print(
        "Total test cases:",
        len(dataset["test_cases"]),
    )

    results = await run_evaluation(
        config,
        dataset,
    )

    correct_predictions = sum(
        1
        for result in results
        if result.get("category_match") == 1
    )

    total_predictions = len(results)

    accuracy = (
        correct_predictions / total_predictions * 100
        if total_predictions > 0
        else 0
    )

    errors = [
        result
        for result in results
        if (
            result.get("status") == "ERROR"
            or result.get("error") is not None
        )
    ]

    print("\n--- Evaluation Summary ---")

    print("Correct predictions:", correct_predictions)
    print("Total evaluated:", total_predictions)
    print(f"Classification accuracy: {accuracy:.2f}%")
    print("Evaluation errors:", len(errors))

    print("\n--- Individual Test Results ---")

    for result in results:
        print("Test ID:", result["test_id"])
        print(
            "Expected category:",
            result["expected_category"],
        )
        print(
            "Predicted category:",
            result["predicted_category"],
        )
        print(
            "Predicted summary:",
            result["predicted_summary"],
        )
        print("Raw output:", result["raw_output"])
        print(
            "Category match:",
            result.get("category_match", 0),
        )
        print(
            "Latency (ms):",
            result.get("latency_ms"),
        )
        print(
            "Token usage:",
            result.get("token_usage"),
        )
        print("Error:", result.get("error"))
        print("Status:", result["status"])
        print("-" * 40)

    expected_count = len(dataset["test_cases"])

    if total_predictions != expected_count:
        raise RuntimeError(
            "Incomplete evaluation: "
            f"expected {expected_count} cases, "
            f"received {total_predictions}."
        )

    if errors:
        failed_ids = [
            str(result["test_id"])
            for result in errors
        ]

        raise RuntimeError(
            "Evaluation contains API or parsing errors "
            f"for test cases: {', '.join(failed_ids)}. "
            "Fix the errors and rerun the evaluation."
        )

    print(
        "\nEvaluation completed successfully "
        "without API or parsing errors."
    )


if __name__ == "__main__":
    asyncio.run(main())
