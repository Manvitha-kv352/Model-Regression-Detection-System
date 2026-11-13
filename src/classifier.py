
import json
import time


async def classify_email(email, config, client):
    start_time = time.perf_counter()

    response = await client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": email}
        ],
        temperature=config.temperature,
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000, 2
    )

    raw_output = response.choices[0].message.content or ""

    try:
        parsed_output = json.loads(raw_output)
        error = None

    except json.JSONDecodeError as exc:
        parsed_output = None
        error = f"Invalid JSON: {exc}"

    usage = response.usage

    token_usage = {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None
    }

    return {
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "error": error,
        "latency_ms": latency_ms,
        "token_usage": token_usage
    }
