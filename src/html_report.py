
import json
import html
from pathlib import Path

REPORT_PATH = Path(
    "evaluations/v1_dataset_v3_scored_report.json"
)

OUTPUT_PATH = Path(
    "reports/evaluation_dashboard.html"
)


def escape(value):
    return html.escape(str(value))


def format_metric(value, suffix=""):
    if value is None:
        return "Not recorded"
    return f"{value}{suffix}"


def main():
    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    summary = report["summary"]
    results = report["results"]

    difficulty_rows = ""
    for difficulty, accuracy in summary[
        "difficulty_accuracy"
    ].items():
        difficulty_rows += f"""
        <tr>
            <td>{escape(difficulty.title())}</td>
            <td>
                <div class="bar-track">
                    <div class="bar-fill"
                         style="width:{float(accuracy)}%">
                    </div>
                </div>
            </td>
            <td>{float(accuracy):.2f}%</td>
        </tr>
        """

    category_rows = ""
    for category, values in summary[
        "accuracy_by_category"
    ].items():
        category_rows += f"""
        <tr>
            <td>{escape(category.title())}</td>
            <td>{values["correct"]}/{values["total"]}</td>
            <td>{values["accuracy_percent"]:.2f}%</td>
        </tr>
        """

    result_rows = ""
    for result in results:
        scores = result["scores"]
        correct = scores["category_match"] == 1

        status = "PASS" if correct else "FAIL"
        status_class = "pass" if correct else "fail"

        usage = scores.get("token_usage")
        tokens = (
            usage.get("total_tokens")
            if isinstance(usage, dict)
            else None
        )

        latency = scores.get("latency_ms")

        result_rows += f"""
        <tr>
            <td>{escape(result["test_id"])}</td>
            <td>{escape(result.get("difficulty") or "Unknown")}</td>
            <td>{escape(result["expected_category"])}</td>
            <td>{escape(result.get("predicted_category"))}</td>
            <td>
                <span class="status {status_class}">
                    {status}
                </span>
            </td>
            <td>{escape(format_metric(scores.get("summary_relevance")))}</td>
            <td>{escape(format_metric(latency, " ms"))}</td>
            <td>{escape(format_metric(tokens))}</td>
        </tr>
        """

    failed_count = sum(
        result["scores"]["category_match"] == 0
        for result in results
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Regression Detection Dashboard</title>
<style>
:root {{
    --bg: #0b1020;
    --surface: #151d32;
    --surface2: #1d2942;
    --text: #eef2ff;
    --muted: #9eacc5;
    --accent: #8b9dff;
    --success: #34d399;
    --danger: #fb7185;
    --border: #2a3651;
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, Segoe UI, Arial, sans-serif;
    line-height: 1.5;
}}
.container {{
    max-width: 1250px;
    margin: auto;
    padding: 42px 24px;
}}
.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 32px;
}}
.eyebrow {{
    color: var(--accent);
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}}
h1 {{ margin: 8px 0; font-size: 32px; }}
h2 {{ font-size: 19px; margin: 0 0 20px; }}
.subtitle {{ color: var(--muted); margin: 0; }}
.pill {{
    background: var(--surface2);
    padding: 10px 16px;
    border-radius: 24px;
    color: var(--accent);
    font-size: 13px;
}}
.metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}}
.card, .panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
}}
.card .value {{
    font-size: 32px;
    font-weight: 750;
    margin-bottom: 6px;
}}
.card .label {{ color: var(--muted); font-size: 13px; }}
.panels {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
}}
.table-wrap {{ overflow-x: auto; }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
th, td {{
    padding: 13px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}}
th {{
    color: var(--muted);
    font-weight: 600;
}}
.bar-track {{
    background: var(--surface2);
    height: 9px;
    min-width: 100px;
    border-radius: 8px;
    overflow: hidden;
}}
.bar-fill {{
    background: var(--accent);
    height: 100%;
    border-radius: 8px;
}}
.status {{
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}}
.pass {{
    color: var(--success);
    background: #123c34;
}}
.fail {{
    color: var(--danger);
    background: #44202d;
}}
.note {{
    margin-top: 16px;
    color: var(--muted);
    font-size: 12px;
}}
.footer {{
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    margin-top: 36px;
}}
@media (max-width: 850px) {{
    .metrics {{ grid-template-columns: repeat(2, 1fr); }}
    .panels {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 480px) {{
    .metrics {{ grid-template-columns: 1fr; }}
    h1 {{ font-size: 25px; }}
}}
</style>
</head>
<body>
<div class="container">

    <header class="header">
        <div>
            <div class="eyebrow">AI QUALITY ENGINEERING</div>
            <h1>Model Regression Detection</h1>
            <p class="subtitle">
                Customer Support LLM · Evaluation Dashboard
            </p>
        </div>
        <div class="pill">
            Prompt {escape(report["prompt_version"])}
            · Dataset {escape(report["dataset_version"])}
        </div>
    </header>

    <section class="metrics">
        <div class="card">
            <div class="value">{summary["category_accuracy"]:.2f}%</div>
            <div class="label">Category Accuracy</div>
        </div>
        <div class="card">
            <div class="value">{escape(format_metric(summary["average_summary_relevance"]))}/5</div>
            <div class="label">Average Summary Relevance</div>
        </div>
        <div class="card">
            <div class="value">{summary["total_cases"]}</div>
            <div class="label">Total Test Cases</div>
        </div>
        <div class="card">
            <div class="value" style="color:var(--danger)">{failed_count}</div>
            <div class="label">Category Mismatches</div>
        </div>
    </section>

    <section class="panels">
        <div class="panel">
            <h2>Accuracy by Difficulty</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Difficulty</th>
                            <th>Accuracy</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody>{difficulty_rows}</tbody>
                </table>
            </div>
        </div>
        <div class="panel">
            <h2>Accuracy by Category</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Correct</th>
                            <th>Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>{category_rows}</tbody>
                </table>
            </div>
        </div>
    </section>

    <section class="panel">
        <h2>Individual Test Results</h2>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Test ID</th>
                        <th>Difficulty</th>
                        <th>Expected</th>
                        <th>Predicted</th>
                        <th>Status</th>
                        <th>Summary /5</th>
                        <th>Latency</th>
                        <th>Tokens</th>
                    </tr>
                </thead>
                <tbody>{result_rows}</tbody>
            </table>
        </div>
        <p class="note">
            Source: cached Prompt v1 evaluation rescored using Dataset v3.
            Missing historical measurements are shown as Not recorded.
            This dashboard does not make API requests.
        </p>
    </section>

    <footer class="footer">
        Model Regression Detection System · Offline Evaluation Report
    </footer>

</div>
</body>
</html>"""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        html_content,
        encoding="utf-8"
    )

    print("HTML dashboard generated successfully!")
    print("Location:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
