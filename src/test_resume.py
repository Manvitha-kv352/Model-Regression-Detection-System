
import asyncio
from unittest.mock import patch

from evaluation_runner import (
    config,
    dataset,
    create_fingerprint,
    save_results,
    run_evaluation
)


async def test_resume():
    fingerprint = create_fingerprint(config, dataset)

    checkpoint_path = "evaluations/v1_dataset_v2_checkpoint.json"

    # Preserve an existing real checkpoint instead of overwriting it.
    # This test uses the runner's checkpoint path, so only proceed
    # when that file does not already exist.
    import os

    if os.path.exists(checkpoint_path):
        print("Checkpoint already exists. Test stopped to protect it.")
        return

    fake_results = [
        {
            "test_id": case["id"],
            "expected_category": case["expected_output"]["category"],
            "predicted_category": case["expected_output"]["category"],
            "predicted_summary": "Simulated result",
            "raw_output": '{"category":"test"}',
            "error": None,
            "status": "PASS"
        }
        for case in dataset["test_cases"][:6]
    ]

    try:
        save_results(fake_results, checkpoint_path, fingerprint)

        # If the runner tries to create a real Groq client,
        # this test fails immediately.
        with patch(
            "evaluation_runner.AsyncGroq",
            side_effect=AssertionError("Unexpected Groq API call")
        ):
            results = await run_evaluation(config, dataset)

        assert len(results) == 6
        assert all(result["status"] == "PASS" for result in results)

        print("PASS: Runner reused all 6 saved results.")
        print("PASS: No Groq client was created.")

    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


if __name__ == "__main__":
    asyncio.run(test_resume())
