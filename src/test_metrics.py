
import asyncio
import os

from dotenv import load_dotenv
from groq import AsyncGroq

from classifier import classify_email
from config import load_prompt_config


load_dotenv()


async def main():
    config = load_prompt_config("prompts/v1.yaml")

    email = (
        "Hi, I was charged twice for my subscription "
        "this month. Please refund the extra payment."
    )

    async with AsyncGroq(
        api_key=os.getenv("GROQ_API_KEY")
    ) as client:

        result = await classify_email(
            email,
            config,
            client
        )

    print("\n--- Fresh Request Metrics ---")
    print("Parsed output:", result["parsed_output"])
    print("Latency (ms):", result["latency_ms"])
    print("Token usage:", result["token_usage"])
    print("Error:", result["error"])

    assert (
        isinstance(result["latency_ms"], (int, float))
        and result["latency_ms"] > 0
    ), "Latency measurement missing"

    usage = result["token_usage"]

    assert (
        isinstance(usage["prompt_tokens"], int)
        and isinstance(usage["completion_tokens"], int)
        and isinstance(usage["total_tokens"], int)
    ), "Token usage missing"

    print("\nMetrics validation: PASSED")


if __name__ == "__main__":
    asyncio.run(main())
