
import asyncio
import os

from dotenv import load_dotenv
from groq import AsyncGroq

from classifier import classify_email
from config import load_prompt_config

load_dotenv()

config = load_prompt_config("prompts/v1.yaml")

email = "I was changed my password still its showing incorrect."


async def main():
    async with AsyncGroq(api_key=os.getenv("GROQ_API_KEY")) as client:
        result = await classify_email(email, config, client)

        print("Category:", result["category"])
        print("Summary:", result["summary"])


asyncio.run(main())
