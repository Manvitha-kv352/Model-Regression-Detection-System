"""Offline category-regression comparison for two saved evaluation checkpoints.

Usage:
 python src/regression_comparator.py --baseline evaluations/old.json --candidate evaluations/new.json --baseline-dataset datasets/golden_dataset_v3.json --candidate-dataset datasets/golden_dataset_v3.json --output evaluations/regression_diff.json

Dataset files are supplied separately because older checkpoints only store a
prompt+dataset fingerprint, not a dataset snapshot. For rigorous provenance,
archive the exact dataset used by EACH run and supply those snapshots here.
"""
import argparse
import json
import os
from pathlib import Path

VALID_STATUSES = {"PASS", "FAIL", "ERROR"}


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def normalize_dataset(path):
    data = read_json(path)
    cases = data if isinstance(data, list) else data.get("test_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Dataset has no test_cases: {path}")
    indexed = {}
    for case in cases:
        test_id = case["id"]
        test_id = f"TC{test_id:03d}" if isinstance(test_id, int) else str(test_id)
        expected = case.get("expected_output")
        category = expected.get("category") if isinstance(expected, dict) else case.get("expected_category")
        email = case.get("input", case.get("email"))
        if not isinstance(email, str) or not isinstance(category, str):
            raise ValueError(f"Invalid input/category in {path}, {test_id}")
        if test_id in indexed:
            raise ValueError(f"Duplicate dataset ID: {test_id}")
        indexed[test_id] = {"input": email, "expected_category": category}
    return indexed


def load_checkpoint(path):
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError(f"Checkpoint must contain a results list: {path}")
    if not isinstance(data.get("fingerprint"), str):
        raise ValueError(f"Checkpoint lacks fingerprint: {path}")
    indexed = {}
    for result in data["results"]:
        test_id = str(result["test_id"])
        if test_id in indexed:
            raise ValueError(f"Duplicate checkpoint ID: {test_id}")
        if result.get("status") not in VALID_STATUSES:
            raise ValueError(f"Invalid status for {test_id}")
        indexed[test_id] = result
    return indexed, data["fingerprint"]


def accuracy(rows):
    return round(100 * sum(r["status"] == "PASS" for r in rows) / len(rows), 2)


def compare(baseline_path, candidate_path, baseline_dataset_path, candidate_dataset_path):
    baseline_dataset = normalize_dataset(baseline_dataset_path)
    candidate_dataset = normalize_dataset(candidate_dataset_path)
    if baseline_dataset != candidate_dataset:
        old_ids, new_ids = set(baseline_dataset), set(candidate_dataset)
        changed = sorted(i for i in old_ids & new_ids if baseline_dataset[i] != candidate_dataset[i])
        raise ValueError(
            "Dataset mismatch: cannot attribute changes to a model/prompt. "
            f"Only baseline IDs: {sorted(old_ids-new_ids)}; "
            f"only candidate IDs: {sorted(new_ids-old_ids)}; "
            f"changed input/label IDs: {changed}"
        )
    old, old_fp = load_checkpoint(baseline_path)
    new, new_fp = load_checkpoint(candidate_path)
    expected_ids = set(baseline_dataset)
    for label, rows in (("baseline", old), ("candidate", new)):
        missing, extra = expected_ids - set(rows), set(rows) - expected_ids
        if missing or extra:
            raise ValueError(f"{label} is not a full matching run. Missing: {sorted(missing)}; extra: {sorted(extra)}")
        for test_id, row in rows.items():
            if row.get("expected_category") != baseline_dataset[test_id]["expected_category"]:
                raise ValueError(f"{label} checkpoint expected label disagrees with supplied dataset for {test_id}")
            # Status must match recorded category prediction; errors are never passes.
            correct = row.get("error") is None and row.get("predicted_category") == row["expected_category"]
            if row["status"] == "PASS" and not correct:
                raise ValueError(f"{label} inconsistent PASS for {test_id}")
            if row["status"] == "FAIL" and correct:
                raise ValueError(f"{label} inconsistent FAIL for {test_id}")

    ids = sorted(expected_ids)
    regressions, improvements, unchanged, errors = [], [], [], []
    categories = sorted({v["expected_category"] for v in baseline_dataset.values()})
    per_category = {}
    for category in categories:
        subset = [i for i in ids if baseline_dataset[i]["expected_category"] == category]
        before = accuracy([old[i] for i in subset])
        after = accuracy([new[i] for i in subset])
        per_category[category] = {"cases": len(subset), "baseline_accuracy": before,
                                  "candidate_accuracy": after, "delta_pp": round(after-before, 2)}
    for test_id in ids:
        a, b = old[test_id], new[test_id]
        item = {"test_id": test_id, "category": baseline_dataset[test_id]["expected_category"],
                "baseline_status": a["status"], "candidate_status": b["status"],
                "baseline_prediction": a.get("predicted_category"),
                "candidate_prediction": b.get("predicted_category")}
        if "ERROR" in (a["status"], b["status"]):
            errors.append(item)  # Do not mislabel infrastructure failures as regressions.
        elif a["status"] == "PASS" and b["status"] == "FAIL":
            regressions.append(item)
        elif a["status"] == "FAIL" and b["status"] == "PASS":
            improvements.append(item)
        else:
            unchanged.append(item)
    before = accuracy(list(old.values()))
    after = accuracy(list(new.values()))
    return {
        "baseline_checkpoint": str(baseline_path), "candidate_checkpoint": str(candidate_path),
        "baseline_fingerprint": old_fp, "candidate_fingerprint": new_fp,
        "dataset_case_count": len(ids),
        "overall": {"baseline_pass_rate": before, "candidate_pass_rate": after,
                    "delta_pp": round(after-before, 2)},
        "per_category": per_category, "regressions": regressions,
        "improvements": improvements, "unchanged_count": len(unchanged),
        "error_transitions": errors,
        "provenance_note": "Dataset snapshots are user-supplied; legacy checkpoint fingerprints cannot independently prove which dataset snapshot produced each run."
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two complete classification evaluation runs")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--candidate-dataset", required=True)
    parser.add_argument("--output", default="evaluations/regression_diff.json")
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.baseline_dataset, args.candidate_dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, output)
    overall = report["overall"]
    print(f"Overall pass rate: {overall['baseline_pass_rate']:.2f}% -> {overall['candidate_pass_rate']:.2f}% ({overall['delta_pp']:+.2f} pp)")
    for name, values in report["per_category"].items():
        print(f"  {name}: {values['baseline_accuracy']:.2f}% -> {values['candidate_accuracy']:.2f}% ({values['delta_pp']:+.2f} pp)")
    print("Regressions:", [r["test_id"] for r in report["regressions"]])
    print("Improvements:", [r["test_id"] for r in report["improvements"]])
    print("Error transitions:", [r["test_id"] for r in report["error_transitions"]])
    print("Saved:", output)


if __name__ == "__main__":
    main()
