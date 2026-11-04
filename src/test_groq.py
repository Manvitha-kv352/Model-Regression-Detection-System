
import os
from dotenv import load_dotenv
from groq import Groq

from config import load_prompt_config

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# Load our prompt configuration
config = load_prompt_config("prompts/v1.yaml")

# Connect to Groq
client = Groq(api_key=api_key)

# Send a test email to the LLM
response = client.chat.completions.create(
    model=config.model,
    messages=[
        {"role": "system", "content": config.system_prompt},
        {
            "role": "user",
            "content": "I was charged twice for my monthly subscription."
        }
    ],
    temperature=config.temperature,
)

print(response.choices[0].message.content)
