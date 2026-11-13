import json
from pathlib import Path

source = Path('datasets/golden_dataset_v2.json')
target = Path('datasets/golden_dataset_v3.json')
if target.exists():
    raise SystemExit(f'{target} already exists; refusing to overwrite.')
data = json.loads(source.read_text(encoding='utf-8'))
assert data['dataset_version'] == 'v2'
assert len(data['test_cases']) == 30
assert len({case['id'] for case in data['test_cases']}) == 30
cases = {case['id']: case for case in data['test_cases']}
assert cases['TC002']['expected_output']['category'] == 'technical'
assert 'invoice' in cases['TC010']['notes'].lower()
data['dataset_version'] = 'v3'
data['description'] = 'Customer support classifier golden dataset, routing policy v1 (customer requested action first)'
data['routing_policy_version'] = 'v1'
cases['TC002']['expected_output']['category'] = 'billing'
cases['TC002']['notes'] = 'Cancellation is the explicit requested action; app crashes are the technical cause.'
cases['TC010']['notes'] = 'Tests delayed promised refund, previous support commitment, and refund status request.'
for case in data['test_cases']:
    case['routing_metadata'] = {
        'label_review_status': 'policy_reviewed',
        'secondary_categories': ['technical'] if case['id'] == 'TC002' else [],
        'mixed_intent': case['id'] == 'TC002',
    }
assert data['total_cases'] == len(data['test_cases'])
target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Created', target, 'with', len(data['test_cases']), 'cases')
