
"""Offline run history, rolling drift and optional Slack webhook alerts.

Run from project root:
    python src/monitoring.py --record
    python src/monitoring.py --send-slack

No LLM calls.

Set SLACK_WEBHOOK_URL in the environment before sending alerts.
A hosted HTTPS report URL is optional.
"""

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_CONFIG = "config/monitoring.json"
DEFAULT_HISTORY = "evaluations/run_history.json"
DEFAULT_GATE = "evaluations/quality_gate_report.json"
DEFAULT_CANDIDATE = "evaluations/v2_dataset_v3_30_checkpoint.json"
DEFAULT_BASELINE = "evaluations/v1_dataset_v3_30_checkpoint.json"
DEFAULT_OUTPUT = "evaluations/monitoring_report.json"


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")

    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    tmp.replace(p)


def accuracy(checkpoint):
    rows = checkpoint["results"]

    if not rows:
        raise ValueError("Checkpoint has no results")

    ids = [str(r["test_id"]) for r in rows]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate test IDs in checkpoint")

    if any(
        r.get("status") not in ("PASS", "FAIL", "ERROR")
        for r in rows
    ):
        raise ValueError("Unknown test status")

    passed = sum(
        r.get("status") == "PASS"
        and r.get("error") is None
        and r.get("predicted_category") == r.get("expected_category")
        for r in rows
    )

    return round(100 * passed / len(rows), 6), len(rows)


def evaluate_drift(history, window, threshold, min_runs):
    if window < 2 or min_runs < window:
        raise ValueError(
            "Require window >= 2 and min_runs >= window"
        )

    if not 0 <= threshold <= 100:
        raise ValueError(
            "Drift threshold must be between 0 and 100"
        )

    if len(history) < min_runs:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "runs_available": len(history),
            "runs_required": min_runs,
            "moving_average": None,
            "threshold": threshold
        }

    recent = history[-window:]

    # Compare only runs using the same dataset and case count.
    keys = {
        (r["dataset_id"], r["total_cases"])
        for r in recent
    }

    if len(keys) != 1:
        return {
            "status": "INCOMPARABLE_RUNS",
            "moving_average": None,
            "threshold": threshold,
            "reason": (
                "Recent runs have different dataset "
                "identifiers or case counts"
            )
        }

    avg = sum(
        float(r["accuracy_percent"])
        for r in recent
    ) / window

    return {
        "status": (
            "SLOW_DRIFT"
            if avg < threshold
            else "STABLE"
        ),
        "moving_average": round(avg, 4),
        "threshold": threshold,
        "window": window,
        "run_ids": [r["run_id"] for r in recent]
    }


def send_slack(webhook, message):
    data = json.dumps(message).encode("utf-8")

    request = urllib.request.Request(
        webhook,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=12
    ) as response:

        body = response.read().decode(
            "utf-8",
            errors="replace"
        )

        if response.status != 200 or body.strip() != "ok":
            raise RuntimeError(
                f"Slack returned HTTP {response.status}: "
                f"{body[:200]}"
            )


def validate_report_url(url):
    """A report URL is optional, but must use HTTPS if provided."""

    if not url:
        return

    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "Report URL must be a valid HTTPS URL."
        )


def build_slack_payload(
    status,
    headline,
    severity,
    drift,
    regressions,
    improvements,
    report_url=None
):
    """Build a Slack alert with an optional report link."""

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"LLM Quality Gate: {status}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{headline}*\n"
                    f"*Gate:* {severity}\n"
                    f"*Drift:* {drift['status']}\n"
                    f"*Regressions:* {regressions}\n"
                    f"*Improvements:* {improvements}"
                )
            }
        }
    ]

    if report_url:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"<{report_url}|"
                    "Open full HTML diff report>"
                )
            }
        })
    else:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        "HTML report is available locally. "
                        "No hosted report link configured."
                    )
                }
            ]
        })

    return {
        "text": (
            f"LLM regression monitor: "
            f"{status} — {headline}"
        ),
        "blocks": blocks
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG
    )

    parser.add_argument(
        "--history",
        default=DEFAULT_HISTORY
    )

    parser.add_argument(
        "--gate",
        default=DEFAULT_GATE
    )

    parser.add_argument(
        "--candidate",
        default=DEFAULT_CANDIDATE
    )

    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE
    )

    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help=(
            "Import baseline checkpoint once; "
            "original timestamp unknown"
        )
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--run-id",
        help="Stable unique run ID; reuse when retrying"
    )

    parser.add_argument(
        "--prompt-version",
        default="v2"
    )

    parser.add_argument(
        "--model",
        default="unknown (supply --model)"
    )

    parser.add_argument(
        "--dataset-id",
        default="v3"
    )

    parser.add_argument(
        "--report-url",
        help="Optional hosted HTTPS HTML report URL"
    )

    parser.add_argument(
        "--record",
        action="store_true",
        help="Record candidate run in history"
    )

    parser.add_argument(
        "--send-slack",
        action="store_true",
        help="Send alert via SLACK_WEBHOOK_URL"
    )

    args = parser.parse_args()

    validate_report_url(args.report_url)

    config = load(args.config)
    gate = load(args.gate)
    candidate = load(args.candidate)

    score, total = accuracy(candidate)

    if abs(
        score - float(gate["candidate_accuracy_percent"])
    ) > 0.02:
        raise ValueError(
            "Quality gate does not match candidate checkpoint; "
            "regenerate it first"
        )

    history = (
        load(args.history)
        if Path(args.history).exists()
        else []
    )

    if not isinstance(history, list):
        raise ValueError(
            "History must be a JSON list"
        )

    if args.record_baseline:
        base = load(args.baseline)
        base_score, base_total = accuracy(base)

        if abs(
            base_score - float(gate["baseline_accuracy_percent"])
        ) > 0.02:
            raise ValueError(
                "Baseline checkpoint and quality gate disagree"
            )

        baseline_id = (
            f"{Path(args.baseline).stem}:"
            f"{base.get('fingerprint', 'unknown')}"
        )

        if not any(
            r["run_id"] == baseline_id
            for r in history
        ):
            history.append({
                "run_id": baseline_id,
                "timestamp_utc": None,
                "prompt_version": "v1",
                "model": args.model,
                "dataset_id": args.dataset_id,
                "total_cases": base_total,
                "accuracy_percent": base_score,
                "checkpoint": args.baseline
            })

            save(args.history, history)

            print(
                "Imported baseline "
                "(original run timestamp not recorded)"
            )

    if args.record:
        run_id = args.run_id or (
            f"{Path(args.candidate).stem}:"
            f"{candidate.get('fingerprint', 'unknown')}"
        )

        entry = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "prompt_version": args.prompt_version,
            "model": args.model,
            "dataset_id": args.dataset_id,
            "total_cases": total,
            "accuracy_percent": score,
            "checkpoint": args.candidate
        }

        existing = next(
            (
                r for r in history
                if r["run_id"] == run_id
            ),
            None
        )

        if existing:
            if (
                existing["accuracy_percent"] != score
                or existing["total_cases"] != total
            ):
                raise ValueError(
                    "Existing run ID has conflicting results"
                )

            print(
                "Run already recorded; history unchanged:",
                run_id
            )

        else:
            history.append(entry)
            save(args.history, history)
            print("Recorded run:", run_id)

    drift = evaluate_drift(
        history,
        int(config["rolling_window"]),
        float(config["drift_threshold_percent"]),
        int(config["minimum_runs"])
    )

    severity = gate["severity"]

    if severity == "CRITICAL":
        status = "FAIL"

    elif (
        severity in ("WARNING", "REVIEW")
        or drift["status"] == "SLOW_DRIFT"
    ):
        status = "WARN"

    else:
        status = "PASS"

    regressions = len(gate["regressions"])
    improvements = len(gate.get("improvements", []))

    baseline = float(
        gate["baseline_accuracy_percent"]
    )

    delta = score - baseline

    # Avoid displaying negative zero due to floating-point rounding.
    if abs(delta) < 0.005:
        delta = 0.0

    headline = (
        f"{regressions} regression(s), "
        f"accuracy {baseline:.2f}% → {score:.2f}% "
        f"({delta:+.2f} pp)"
    )

    report = {
        "status": status,
        "quality_gate": severity,
        "headline": headline,
        "drift": drift,
        "run_count": len(history),
        "report_url": args.report_url,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    save(args.output, report)

    print(
        "Alert status:",
        status,
        "|",
        headline
    )

    print(
        "Drift:",
        drift["status"],
        "| moving average:",
        drift.get("moving_average")
    )

    print("Saved:", args.output)

    if args.send_slack:
        webhook = os.getenv("SLACK_WEBHOOK_URL")

        if not webhook:
            raise ValueError(
                "Set SLACK_WEBHOOK_URL in the environment; "
                "never commit it"
            )

        if (
            status == "PASS"
            and not config.get(
                "send_pass_notifications",
                False
            )
        ):
            print(
                "Slack skipped: PASS notifications disabled"
            )

        else:
            payload = build_slack_payload(
                status=status,
                headline=headline,
                severity=severity,
                drift=drift,
                regressions=regressions,
                improvements=improvements,
                report_url=args.report_url
            )

            send_slack(webhook, payload)

            print("Slack alert sent")


if __name__ == "__main__":
    main()
