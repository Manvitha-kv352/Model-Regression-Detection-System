
import asyncio
from unittest.mock import patch

from groq import RateLimitError

from evaluation_runner import run_evaluation, config, dataset


async def fake_classify_email(email, config, client):
    raise RateLimitError(
        "Simulated rate limit",
        response=type(
            "FakeResponse",
            (),
            {
                "status_code": 429,
                "headers": {},
                "request": None
            }
        )(),
        body=None
    )


async def main():
    test_dataset = {
        **dataset,
        "test_cases": dataset["test_cases"][:1]
    }

    with patch("evaluation_runner.classify_email", fake_classify_email):
        results = await run_evaluation(config, test_dataset)

    print("Test ID:", results[0]["test_id"])
    print("Predicted category:", results[0]["predicted_category"])
    print("Error recorded:", results[0]["error"] is not None)


if __name__ == "__main__":
    asyncio.run(main())
