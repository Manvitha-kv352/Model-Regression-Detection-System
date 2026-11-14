"""Offline paired regression quality gate. Run from the project root.

python src/quality_gate.py
python src/quality_gate.py --config config/quality_gate.json
"""
import argparse
import json
import math
from pathlib import Path

DEFAULTS = {
    "warning_delta_percent_points": 3.0,
    "critical_delta_percent_points": 8.0,
    "significance_alpha": 0.05,
    "flag_any_regression": True,
}


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(path):
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError(f"Invalid checkpoint: {path}")
    rows = {}
    for row in data["results"]:
        key = str(row["test_id"])
        if key in rows:
            raise ValueError(f"Duplicate test ID {key} in {path}")
        rows[key] = row
    return rows


def passed(row):
    return (row.get("status") == "PASS"
            and row.get("error") is None
            and row.get("predicted_category") == row.get("expected_category"))


def mcnemar_exact(better, worse):
    """Exact two-sided McNemar p-value for discordant paired outcomes."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    # Sum small binomial tail with integer arithmetic; no scipy dependency.
    numerator = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * numerator / (2 ** n))


def assess(baseline, candidate, config):
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("Baseline and candidate must contain the same nonempty test IDs")
    improvements, regressions, unchanged = [], [], []
    for key in sorted(baseline):
        old, new = baseline[key], candidate[key]
        if old.get("expected_category") != new.get("expected_category"):
            raise ValueError(f"Expected category changed for {key}; use the same golden dataset")
        if passed(old) and not passed(new):
            regressions.append(key)
        elif not passed(old) and passed(new):
            improvements.append(key)
        else:
            unchanged.append(key)
    total = len(baseline)
    old_ok = sum(passed(r) for r in baseline.values())
    new_ok = sum(passed(r) for r in candidate.values())
    delta_pp = 100.0 * (new_ok - old_ok) / total
    drop_pp = max(0.0, -delta_pp)
    warning = config["warning_delta_percent_points"]
    critical = config["critical_delta_percent_points"]
    if drop_pp > critical:
        severity = "CRITICAL"
    elif drop_pp > warning:
        severity = "WARNING"
    elif regressions and config["flag_any_regression"]:
        severity = "REVIEW"
    else:
        severity = "PASS"
    p_value = mcnemar_exact(len(improvements), len(regressions))
    return {
        "severity": severity,
        "release_decision": "REVIEW_REQUIRED" if severity != "PASS" else "PASS",
        "baseline_accuracy_percent": round(100 * old_ok / total, 4),
        "candidate_accuracy_percent": round(100 * new_ok / total, 4),
        "delta_percentage_points": round(delta_pp, 4),
        "accuracy_drop_percentage_points": round(drop_pp, 4),
        "total_paired_cases": total,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged_case_count": len(unchanged),
        "statistics": {
            "test": "exact_two_sided_mcnemar",
            "discordant_pairs": len(improvements) + len(regressions),
            "p_value": round(p_value, 8),
            "alpha": config["significance_alpha"],
            "statistically_significant": p_value < config["significance_alpha"],
            "note": "A non-significant result does not prove equivalence. Small samples have limited power."
        },
        "thresholds": config,
        "notes": [
            "Thresholds measure accuracy decreases in percentage points, not relative percent change.",
            "WARNING/CRITICAL use strict greater-than comparisons; equal to threshold does not trigger.",
            "McNemar tests net differences in paired pass/fail outcomes, not individual category changes.",
            "Use identical dataset inputs and labels across both runs; matching IDs and labels alone cannot verify identical inputs."
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Offline regression significance and quality gate")
    parser.add_argument("--baseline", default="evaluations/v1_dataset_v3_30_checkpoint.json")
    parser.add_argument("--candidate", default="evaluations/v2_dataset_v3_30_checkpoint.json")
    parser.add_argument("--config", default="config/quality_gate.json")
    parser.add_argument("--output", default="evaluations/quality_gate_report.json")
    args = parser.parse_args()
    config = dict(DEFAULTS)
    if Path(args.config).exists():
        config.update(load_json(args.config))
    for key in ("warning_delta_percent_points", "critical_delta_percent_points", "significance_alpha"):
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool):
            raise ValueError(f"{key} must be numeric")
    if not (0 <= config["warning_delta_percent_points"] < config["critical_delta_percent_points"] <= 100):
        raise ValueError("Require 0 <= warning < critical <= 100")
    if not 0 < config["significance_alpha"] < 1:
        raise ValueError("significance_alpha must be between 0 and 1")
    if not isinstance(config["flag_any_regression"], bool):
        raise ValueError("flag_any_regression must be boolean")
    result = assess(load_checkpoint(args.baseline), load_checkpoint(args.candidate), config)
    result["files"] = {"baseline": args.baseline, "candidate": args.candidate}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Baseline: {result['baseline_accuracy_percent']:.2f}%")
    print(f"Candidate: {result['candidate_accuracy_percent']:.2f}%")
    print(f"Delta: {result['delta_percentage_points']:+.2f} percentage points")
    print(f"Regressions: {result['regressions']}")
    print(f"Improvements: {result['improvements']}")
    print(f"McNemar exact p-value: {result['statistics']['p_value']:.6f}")
    print(f"Statistically significant: {result['statistics']['statistically_significant']}")
    print(f"Quality gate: {result['severity']} ({result['release_decision']})")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
