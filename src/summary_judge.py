
import json


JUDGE_SYSTEM_PROMPT = """
You are an impartial evaluator of customer-support email summaries.

Compare the original customer email with the generated summary.

Rate summary relevance using this scale:
1 = Incorrect or unrelated
2 = Major information missing or misleading
3 = Partially accurate, important details missing
4 = Accurate, minor omissions
5 = Fully accurate and faithful to the email

Rules:
- Judge only the summary, not the category.
- Penalize invented information.
- Consider whether the main customer issue is preserved.
- Do not reward length unnecessarily.
- Return only valid JSON with:
  "score": an integer from 1 to 5
  "reason": a short explanation
"""


async def judge_summary(
    original_email,
    generated_summary,
    client,
    model
):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps({
                    "original_email": original_email,
                    "generated_summary": generated_summary
                })
            }
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content or ""

    try:
        parsed = json.loads(raw_output)

        score = parsed.get("score")
        reason = parsed.get("reason")

        if (
            type(score) is not int
            or not 1 <= score <= 5
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("Invalid judge score or reason")

        return {
            "summary_relevance": score,
            "judge_reason": reason,
            "judge_error": None
        }

    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        return {
            "summary_relevance": None,
            "judge_reason": None,
            "judge_error": str(exc)
        }
