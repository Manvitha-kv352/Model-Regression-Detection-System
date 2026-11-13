
ALLOWED_CATEGORIES = {
    "billing",
    "technical",
    "account",
    "general",
}


def validate_schema(output):
    """Validate the classifier's JSON structure."""
    if not isinstance(output, dict):
        return False

    if set(output.keys()) != {"category", "summary"}:
        return False

    if output["category"] not in ALLOWED_CATEGORIES:
        return False

    if not isinstance(output["summary"], str):
        return False

    if not output["summary"].strip():
        return False

    return True


def score_case(test_case, classifier_result, judgment=None):
    """Score one classifier prediction against a golden case."""

    parsed = classifier_result.get("parsed_output")

    # Support previously saved classifier results
    if parsed is None:
        raw = classifier_result.get("raw_output")
        if isinstance(raw, str):
            import json
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None

    schema_valid = validate_schema(parsed)

    expected_category = test_case["expected_output"]["category"]

    predicted_category = (
        parsed.get("category")
        if isinstance(parsed, dict)
        else classifier_result.get("predicted_category")
    )

    category_match = int(
        schema_valid
        and predicted_category == expected_category
    )

    relevance = None
    judge_reason = None
    judge_error = None

    if judgment is not None:
        relevance = judgment.get("summary_relevance")
        judge_reason = judgment.get("judge_reason")
        judge_error = judgment.get("judge_error")

    return {
        "test_id": test_case["id"],
        "difficulty": test_case.get("difficulty"),
        "expected_category": expected_category,
        "predicted_category": predicted_category,
        "scores": {
            "schema_valid": schema_valid,
            "category_match": category_match,
            "summary_relevance": relevance,
            "latency_ms": classifier_result.get("latency_ms"),
            "token_usage": classifier_result.get("token_usage"),
        },
        "judge_reason": judge_reason,
        "technical_error": classifier_result.get("error"),
        "judge_error": judge_error,
    }


def summarize_scores(scored_results):
    """Calculate overall and grouped evaluation metrics."""

    total = len(scored_results)

    if total == 0:
        return {}

    correct = sum(
        result["scores"]["category_match"]
        for result in scored_results
    )

    relevance_scores = [
        result["scores"]["summary_relevance"]
        for result in scored_results
        if isinstance(
            result["scores"]["summary_relevance"],
            int
        )
    ]

    grouped = {}

    for result in scored_results:
        difficulty = result["difficulty"] or "unknown"

        if difficulty not in grouped:
            grouped[difficulty] = {
                "total": 0,
                "correct": 0,
            }

        grouped[difficulty]["total"] += 1
        grouped[difficulty]["correct"] += (
            result["scores"]["category_match"]
        )

    difficulty_accuracy = {
        difficulty: round(
            values["correct"] / values["total"] * 100,
            2
        )
        for difficulty, values in grouped.items()
    }

    return {
        "total_cases": total,
        "correct_categories": correct,
        "category_accuracy": round(
            correct / total * 100,
            2
        ),
        "average_summary_relevance": (
            round(
                sum(relevance_scores) / len(relevance_scores),
                2
            )
            if relevance_scores
            else None
        ),
        "difficulty_accuracy": difficulty_accuracy,
    }
