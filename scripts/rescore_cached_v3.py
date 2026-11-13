import json
from pathlib import Path

old = json.loads(Path('datasets/golden_dataset_v2.json').read_text(encoding='utf-8'))
new = json.loads(Path('datasets/golden_dataset_v3.json').read_text(encoding='utf-8'))
checkpoint = json.loads(Path('evaluations/v1_dataset_v2_30_checkpoint.json').read_text(encoding='utf-8'))
assert old['dataset_version'] == 'v2' and new['dataset_version'] == 'v3'
assert len(old['test_cases']) == len(new['test_cases']) == 30
assert isinstance(checkpoint, dict) and isinstance(checkpoint.get('results'), list)
old_by_id = {c['id']: c for c in old['test_cases']}
new_by_id = {c['id']: c for c in new['test_cases']}
assert set(old_by_id) == set(new_by_id)
for test_id in old_by_id:
    assert old_by_id[test_id]['input'] == new_by_id[test_id]['input'], f'Input changed: {test_id}; cannot reuse predictions'
results = checkpoint['results']
assert len(results) == 30
assert len({r['test_id'] for r in results}) == 30
assert {r['test_id'] for r in results} == set(new_by_id)
rows = []
for r in results:
    case = new_by_id[r['test_id']]
    expected = case['expected_output']['category']
    predicted = r.get('predicted_category')
    rows.append({'test_id': r['test_id'], 'expected_category': expected,
                 'predicted_category': predicted, 'category_match': int(r.get('error') is None and predicted == expected),
                 'difficulty': case['difficulty']})
rows.sort(key=lambda x: x['test_id'])
report = {'dataset_version': 'v3', 'routing_policy_version': 'v1',
          'source_checkpoint': 'evaluations/v1_dataset_v2_30_checkpoint.json',
          'note': 'Re-scored cached predictions; no new classifier requests. NOT a new inference checkpoint.',
          'correct': sum(r['category_match'] for r in rows), 'total': len(rows),
          'accuracy_percent': round(100 * sum(r['category_match'] for r in rows) / len(rows), 2),
          'results': rows}
out = Path('evaluations/v1_cached_predictions_rescored_dataset_v3.json')
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    raise SystemExit(f'{out} exists; refusing to overwrite')
out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(f"Re-scored: {report['correct']}/{report['total']} = {report['accuracy_percent']}%")
print('Mismatches:', [r['test_id'] for r in rows if not r['category_match']])
print('Saved', out)
