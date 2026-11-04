import json
import os
from dotenv import load_dotenv
from groq import Groq

from config import load_prompt_config

load_dotenv()

config = load_prompt_config("prompts/v1.yaml")

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)


def classify_email(email):

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": email}
        ],
        temperature=config.temperature,
    )

    result = response.choices[0].message.content
    return json.loads(result)
