
import asyncio
import json
import os

from dotenv import load_dotenv
from groq import AsyncGroq, RateLimitError

from evaluation_runner import (
    config,
    dataset,
    create_fingerprint,
    load_results,
    save_results,
)
from summary_judge import judge_summary


load_dotenv()

# Dataset v3 checkpoints
CLASSIFIER_CHECKPOINT = (
    "evaluations/v1_dataset_v3_30_checkpoint.json"
)

JUDGE_CHECKPOINT = (
    "evaluations/v1_dataset_v3_30_judged.json"
)

BATCH_SIZE = 3
BATCH_DELAY = 20
MAX_RETRIES = 5


async def run_judging():
    fingerprint = create_fingerprint(config, dataset)

    # Load previously saved classifier predictions
    classifier_results = load_results(
        CLASSIFIER_CHECKPOINT,
        fingerprint
    )

    if not classifier_results:
        raise ValueError(
            "No classifier results found. "
            "Check the Dataset v3 checkpoint and fingerprint."
        )

    # Map golden dataset cases by ID
    case_by_id = {
        case["id"]: case
        for case in dataset["test_cases"]
    }

    # Validate saved classifier test IDs
    saved_ids = [
        result["test_id"]
        for result in classifier_results
    ]

    unknown_ids = set(saved_ids) - set(case_by_id)

    if unknown_ids:
        raise ValueError(
            f"Unknown test IDs: {sorted(unknown_ids)}"
        )

    if len(saved_ids) != len(set(saved_ids)):
        raise ValueError(
            "Duplicate test IDs in classifier checkpoint."
        )

    # Judge only cases already evaluated
    selected_cases = [
        case
        for case in dataset["test_cases"]
        if case["id"] in set(saved_ids)
    ]

    # Preserve all classifier metrics
    results_by_id = {
        result["test_id"]: dict(result)
        for result in classifier_results
    }

    # Resume previous judge progress
    previous_results = load_results(
        JUDGE_CHECKPOINT,
        fingerprint
    )

    for result in previous_results:
        test_id = result["test_id"]

        if test_id in results_by_id:
            results_by_id[test_id].update({
                "summary_relevance": result.get(
                    "summary_relevance"
                ),
                "judge_reason": result.get(
                    "judge_reason"
                ),
                "judge_error": result.get(
                    "judge_error"
                )
            })

    # Cases without a valid judge score
    pending_cases = [
        case
        for case in selected_cases
        if not (
            isinstance(
                results_by_id[case["id"]].get(
                    "summary_relevance"
                ),
                int
            )
            and 1 <= results_by_id[case["id"]][
                "summary_relevance"
            ] <= 5
        )
    ]

    print("Prompt version:", config.version)
    print("Dataset version:", dataset["dataset_version"])
    print("Total dataset cases:", len(dataset["test_cases"]))
    print("Saved classifier cases:", len(selected_cases))
    print(
        "Already judged:",
        len(selected_cases) - len(pending_cases)
    )
    print("Remaining to judge:", len(pending_cases))

    if not pending_cases:
        print("All saved summaries are already judged.")
        return

    semaphore = asyncio.Semaphore(3)

    async with AsyncGroq(
        api_key=os.getenv("GROQ_API_KEY")
    ) as client:

        async def judge_one(case):
            test_id = case["id"]
            result = results_by_id[test_id]

            summary = result.get("predicted_summary")

            if not isinstance(summary, str) or not summary.strip():
                return {
                    "test_id": test_id,
                    "summary_relevance": None,
                    "judge_reason": None,
                    "judge_error": "Missing generated summary"
                }

            for attempt in range(MAX_RETRIES):
                try:
                    async with semaphore:
                        judgment = await judge_summary(
                            original_email=case["input"],
                            generated_summary=summary,
                            client=client,
                            model=config.model
                        )

                    return {
                        "test_id": test_id,
                        **judgment
                    }

                except RateLimitError as exc:
                    if attempt == MAX_RETRIES - 1:
                        return {
                            "test_id": test_id,
                            "summary_relevance": None,
                            "judge_reason": None,
                            "judge_error": (
                                f"Rate limit after retries: {exc}"
                            )
                        }

                    wait_time = min(
                        3 * (2 ** attempt),
                        30
                    )

                    print(
                        f"Rate limit for {test_id}. "
                        f"Retrying in {wait_time}s..."
                    )

                    await asyncio.sleep(wait_time)

        # Process judge requests in batches
        for i in range(
            0,
            len(pending_cases),
            BATCH_SIZE
        ):
            batch = pending_cases[i:i + BATCH_SIZE]

            print(
                f"Judging batch {i // BATCH_SIZE + 1}: "
                f"{len(batch)} cases"
            )

            batch_results = await asyncio.gather(
                *[judge_one(case) for case in batch]
            )

            for judgment in batch_results:
                results_by_id[
                    judgment["test_id"]
                ].update(judgment)

            ordered_results = [
                results_by_id[case["id"]]
                for case in selected_cases
            ]

            # Save classifier metrics + judge scores
            save_results(
                ordered_results,
                JUDGE_CHECKPOINT,
                fingerprint
            )

            print(
                "Saved judge progress:",
                JUDGE_CHECKPOINT
            )

            if i + BATCH_SIZE < len(pending_cases):
                await asyncio.sleep(BATCH_DELAY)

    # Final summary
    print("\n--- Multi-Dimensional Scoring Summary ---")

    ordered_results = [
        results_by_id[case["id"]]
        for case in selected_cases
    ]

    valid_scores = [
        result["summary_relevance"]
        for result in ordered_results
        if type(result.get("summary_relevance")) is int
        and 1 <= result["summary_relevance"] <= 5
    ]

    print("Evaluated cases:", len(ordered_results))
    print("Scored summaries:", len(valid_scores))

    if valid_scores:
        print(
            "Average relevance:",
            round(sum(valid_scores) / len(valid_scores), 2),
            "/ 5"
        )

    for result in ordered_results:
        print("\nTest ID:", result["test_id"])
        print(
            "Category match:",
            result.get("category_match")
        )
        print(
            "Summary relevance:",
            result.get("summary_relevance")
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
            "Judge error:",
            result.get("judge_error")
        )
        print("-" * 45)


if __name__ == "__main__":
    asyncio.run(run_judging())
