
import asyncio
import os
from unittest.mock import patch

import evaluation_runner as runner


async def test_partial_resume():
    # Use an isolated checkpoint, never the real one
    checkpoint_path = "evaluations/partial_resume_test.json"

    if os.path.exists(checkpoint_path):
        print("Test checkpoint already exists. Stopping to protect it.")
        return

    original_load = runner.load_results
    original_save = runner.save_results

    fingerprint = runner.create_fingerprint(
        runner.config,
        runner.dataset
    )

    test_cases = runner.dataset["test_cases"][:6]

    # First 3 cases are already completed
    saved_results = [
        {
            "test_id": case["id"],
            "expected_category": case["expected_output"]["category"],
            "predicted_category": case["expected_output"]["category"],
            "predicted_summary": "Previously completed",
            "raw_output": '{"category":"test"}',
            "error": None,
            "status": "PASS"
        }
        for case in test_cases[:3]
    ]

    try:
        original_save(saved_results, checkpoint_path, fingerprint)

        def mock_load(file_path, expected_fingerprint):
            return original_load(
                checkpoint_path,
                expected_fingerprint
            )

        def mock_save(results, file_path, fingerprint):
            return original_save(
                results,
                checkpoint_path,
                fingerprint
            )

        async def mock_evaluate(test_case, config, client):
            return {
                "test_id": test_case["id"],
                "expected_category": test_case["expected_output"]["category"],
                "predicted_category": test_case["expected_output"]["category"],
                "predicted_summary": "New simulated result",
                "raw_output": '{"category":"test"}',
                "error": None,
                "status": "PASS"
            }

        class FakeGroq:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with (
            patch.object(runner, "load_results", side_effect=mock_load),
            patch.object(runner, "save_results", side_effect=mock_save),
            patch.object(runner, "evaluate_case", side_effect=mock_evaluate),
            patch.object(runner, "AsyncGroq", return_value=FakeGroq()),
        ):
            results = await runner.run_evaluation(
                runner.config,
                runner.dataset
            )

        assert len(results) == 6
        assert len({r["test_id"] for r in results}) == 6

        assert all(
            r["predicted_summary"] == "Previously completed"
            for r in results[:3]
        )

        assert all(
            r["predicted_summary"] == "New simulated result"
            for r in results[3:]
        )

        checkpoint = original_load(checkpoint_path, fingerprint)
        assert len(checkpoint) == 6

        print("PASS: 3 previous results were reused.")
        print("PASS: 3 pending cases were evaluated.")
        print("PASS: All 6 results were saved without duplicates.")
        print("PASS: No Groq API calls were made.")

    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


if __name__ == "__main__":
    asyncio.run(test_partial_resume())
