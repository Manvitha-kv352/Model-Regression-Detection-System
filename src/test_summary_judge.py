
import asyncio
import json
import os

from dotenv import load_dotenv
from groq import AsyncGroq

from config import load_prompt_config
from summary_judge import judge_summary


load_dotenv()

config = load_prompt_config("prompts/v1.yaml")

with open(
    "datasets/golden_dataset_v3.json",
    "r",
    encoding="utf-8"
) as file:
    dataset = json.load(file)

case = next(
    case
    for case in dataset["test_cases"]
    if case["id"] == "TC002"
)

original_email = case["input"]

fixtures = [
    {
        "name": "Accurate summary",
        "summary": (
            "Customer wants to cancel their subscription "
            "because the application keeps crashing."
        ),
        "minimum": 5,
        "maximum": 5
    },
    {
        "name": "Incomplete summary",
        "summary": "Customer reports application crashes.",
        "minimum": 3,
        "maximum": 4
    },
    {
        "name": "Wrong customer request",
        "summary": "Customer wants to upgrade their subscription.",
        "minimum": 1,
        "maximum": 2
    },
    {
        "name": "Invented information",
        "summary": (
            "Customer wants to cancel because they "
            "were charged twice and received no refund."
        ),
        "minimum": 1,
        "maximum": 3
    },
    {
        "name": "Unrelated summary",
        "summary": "Customer asks about support working hours.",
        "minimum": 1,
        "maximum": 1
    }
]


async def main():
    passed = 0

    async with AsyncGroq(
        api_key=os.getenv("GROQ_API_KEY")
    ) as client:

        for fixture in fixtures:
            print("\n---", fixture["name"], "---")

            result = await judge_summary(
                original_email=original_email,
                generated_summary=fixture["summary"],
                client=client,
                model=config.model
            )

            score = result["summary_relevance"]

            print("Summary:", fixture["summary"])
            print("Judge score:", score)
            print("Judge reason:", result["judge_reason"])
            print("Judge error:", result["judge_error"])

            expected = (
                score is not None
                and fixture["minimum"] <= score <= fixture["maximum"]
            )

            print("Within expected range:", expected)

            if expected:
                passed += 1

            # Space requests to reduce rate-limit pressure
            await asyncio.sleep(5)

    print("\n==========================")
    print(
        f"Judge calibration passed: "
        f"{passed}/{len(fixtures)}"
    )
    print("==========================")


if __name__ == "__main__":
    asyncio.run(main())
