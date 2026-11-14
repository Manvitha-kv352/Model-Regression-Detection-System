"""Generate a standalone, evidence-backed HTML regression dashboard (no API calls).

Run from project root:
    python src/html_report.py

Use --baseline, --candidate, --dataset and --diff for other evaluation runs.
"""
import argparse
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BASELINE = 'evaluations/v1_dataset_v3_30_checkpoint.json'
DEFAULT_CANDIDATE = 'evaluations/v2_dataset_v3_30_checkpoint.json'
DEFAULT_DATASET = 'datasets/golden_dataset_v3.json'
DEFAULT_DIFF = 'evaluations/regression_diff.json'
DEFAULT_OUTPUT = 'reports/evaluation_dashboard.html'


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def esc(value):
    return html.escape(str(value if value is not None else '—'), quote=True)


def pct(n, d):
    return 100 * n / d if d else 0.0


def metric(value, suffix='', digits=2):
    return 'Not recorded' if not isinstance(value, (int, float)) else f'{value:,.{digits}f}{suffix}'


def checkpoint(path):
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get('results'), list):
        raise ValueError(f'Invalid checkpoint: {path}')
    indexed = {}
    for item in data['results']:
        test_id = str(item['test_id'])
        if test_id in indexed:
            raise ValueError(f'Duplicate test ID in {path}: {test_id}')
        indexed[test_id] = item
    return indexed


def load_cases(path):
    raw = read_json(path)
    cases = raw if isinstance(raw, list) else raw.get('test_cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Dataset has no test cases')
    output = {}
    for case in cases:
        test_id = case['id']
        test_id = f'TC{test_id:03d}' if isinstance(test_id, int) else str(test_id)
        expected = case.get('expected_output')
        category = expected.get('category') if isinstance(expected, dict) else case.get('expected_category')
        if test_id in output:
            raise ValueError(f'Duplicate dataset ID: {test_id}')
        output[test_id] = {
            'email': case.get('input', case.get('email', '')),
            'category': category,
            'difficulty': case.get('difficulty') or case.get('complexity') or 'Not labeled',
            'expected_summary': (expected.get('summary') if isinstance(expected, dict) else case.get('expected_summary')),
        }
    return output


def score(item):
    return item.get('status') == 'PASS' and item.get('error') is None and item.get('predicted_category') == item.get('expected_category')


def run_metrics(rows):
    values = list(rows.values())
    successes = sum(score(r) for r in values)
    latencies = [r.get('latency_ms') for r in values if isinstance(r.get('latency_ms'), (int, float))]
    tokens = [r.get('token_usage', {}).get('total_tokens') for r in values if isinstance(r.get('token_usage'), dict) and isinstance(r['token_usage'].get('total_tokens'), (int, float))]
    judged = [r.get('summary_relevance') for r in values if isinstance(r.get('summary_relevance'), (int, float))]
    return {'correct': successes, 'total': len(values), 'accuracy': pct(successes, len(values)),
            'latency': sum(latencies)/len(latencies) if latencies else None,
            'tokens': sum(tokens) if len(tokens) == len(values) else None,
            'judged': sum(judged)/len(judged) if judged else None,
            'judged_count': len(judged)}


def make_dashboard(baseline_path, candidate_path, dataset_path, diff_path, output_path):
    cases = load_cases(dataset_path)
    before = checkpoint(baseline_path)
    after = checkpoint(candidate_path)
    diff = read_json(diff_path)
    ids = sorted(cases)
    if set(before) != set(ids) or set(after) != set(ids):
        raise ValueError('Checkpoints must both contain exactly the dataset test IDs.')
    for test_id in ids:
        for name, item in [('baseline', before[test_id]), ('candidate', after[test_id])]:
            if item.get('expected_category') != cases[test_id]['category']:
                raise ValueError(f'{name} expected label mismatch: {test_id}')
            if item.get('status') not in ('PASS', 'FAIL', 'ERROR'):
                raise ValueError(f'{name} invalid status: {test_id}')
    # Ensure the comparison report corresponds to these checkpoint files, not an older diff.
    if Path(diff.get('baseline_checkpoint', '')).as_posix() != Path(baseline_path).as_posix() or Path(diff.get('candidate_checkpoint', '')).as_posix() != Path(candidate_path).as_posix():
        raise ValueError('Regression diff references different checkpoint paths. Regenerate regression_diff.json first.')
    if diff.get('dataset_case_count') != len(ids):
        raise ValueError('Regression diff case count does not match dataset.')
    bm, cm = run_metrics(before), run_metrics(after)
    if abs(diff['overall']['baseline_pass_rate'] - round(bm['accuracy'], 2)) > .01 or abs(diff['overall']['candidate_pass_rate'] - round(cm['accuracy'], 2)) > .01:
        raise ValueError('Regression diff accuracy disagrees with checkpoints. Regenerate the diff.')
    regressions = {r['test_id'] for r in diff['regressions']}
    improvements = {r['test_id'] for r in diff['improvements']}
    errors = {r['test_id'] for r in diff['error_transitions']}
    actual_regressions = {i for i in ids if score(before[i]) and not score(after[i]) and i not in errors}
    actual_improvements = {i for i in ids if not score(before[i]) and score(after[i]) and i not in errors}
    if regressions != actual_regressions or improvements != actual_improvements:
        raise ValueError('Regression diff case transitions disagree with checkpoints.')

    category_rows = []
    for category in sorted({cases[i]['category'] for i in ids}):
        subset = [i for i in ids if cases[i]['category'] == category]
        old_rate = pct(sum(score(before[i]) for i in subset), len(subset))
        new_rate = pct(sum(score(after[i]) for i in subset), len(subset))
        delta = new_rate-old_rate
        delta_class = 'good' if delta > .001 else 'bad' if delta < -.001 else 'muted'
        category_rows.append(f'''<tr><td><strong>{esc(category.title())}</strong><span class="sub">{len(subset)} cases</span></td>
        <td>{old_rate:.2f}%</td><td>{new_rate:.2f}%</td><td class="{delta_class}">{delta:+.2f} pp</td>
        <td><div class="comparison-bars"><div class="bar baseline" style="width:{old_rate:.2f}%"></div><div class="bar candidate" style="width:{new_rate:.2f}%"></div></div></td></tr>''')

    detail_rows = []
    for test_id in ids:
        a, b, case = before[test_id], after[test_id], cases[test_id]
        if test_id in regressions:
            change, change_class = 'Regression', 'bad'
        elif test_id in improvements:
            change, change_class = 'Improvement', 'good'
        elif test_id in errors:
            change, change_class = 'Error transition', 'warn'
        else:
            change, change_class = 'Unchanged', 'muted'
        old_status, new_status = a['status'], b['status']
        def status_class(s):
            return 'good' if s == 'PASS' else 'bad' if s == 'FAIL' else 'warn'
        detail_rows.append(f'''<tr class="case-row" data-filter="{change.lower().replace(' ', '-')}" data-search="{esc(' '.join([test_id, case['email'], case['category'], str(a.get('predicted_category')), str(b.get('predicted_category'))]).lower())}">
          <td><strong>{esc(test_id)}</strong></td><td>{esc(case['category'])}</td>
          <td>{esc(a.get('predicted_category'))}<span class="sub {status_class(old_status)}">{esc(old_status)}</span></td>
          <td>{esc(b.get('predicted_category'))}<span class="sub {status_class(new_status)}">{esc(new_status)}</span></td>
          <td><span class="tag {change_class}">{esc(change)}</span></td>
          <td><details><summary>Inspect evidence</summary><div class="evidence"><strong>Customer email</strong><p>{esc(case['email'])}</p>
          <strong>Expected category</strong><p>{esc(case['category'])}</p>
          <strong>Baseline summary</strong><p>{esc(a.get('predicted_summary'))}</p>
          <strong>Candidate summary</strong><p>{esc(b.get('predicted_summary'))}</p>
          <strong>Baseline latency / tokens</strong><p>{metric(a.get('latency_ms'), ' ms')} / {esc(a.get('token_usage', {}).get('total_tokens') if isinstance(a.get('token_usage'), dict) else None)}</p>
          <strong>Candidate latency / tokens</strong><p>{metric(b.get('latency_ms'), ' ms')} / {esc(b.get('token_usage', {}).get('total_tokens') if isinstance(b.get('token_usage'), dict) else None)}</p>
          <strong>Errors</strong><p>Baseline: {esc(a.get('error'))} | Candidate: {esc(b.get('error'))}</p>
          </div></details></td></tr>''')

    def summary_card(label, value, note, color=''):
        return f'<div class="card"><div class="label">{esc(label)}</div><div class="value {color}">{esc(value)}</div><div class="sub">{esc(note)}</div></div>'
    delta = cm['accuracy'] - bm['accuracy']
    cards = ''.join([
        summary_card('Baseline accuracy', f"{bm['accuracy']:.2f}%", f"{bm['correct']}/{bm['total']} passed"),
        summary_card('Candidate accuracy', f"{cm['accuracy']:.2f}%", f"{cm['correct']}/{cm['total']} passed"),
        summary_card('Overall change', f'{delta:+.2f} pp', 'Percentage points', 'good' if delta > 0 else 'bad' if delta < 0 else ''),
        summary_card('Regressions', str(len(regressions)), 'PASS → FAIL', 'bad' if regressions else 'good'),
        summary_card('Improvements', str(len(improvements)), 'FAIL → PASS', 'good'),
        summary_card('Error transitions', str(len(errors)), 'Investigate separately', 'warn' if errors else ''),
    ])
    metric_rows = f'''<tr><td>Average latency</td><td>{metric(bm['latency'], ' ms')}</td><td>{metric(cm['latency'], ' ms')}</td></tr>
    <tr><td>Total classifier tokens</td><td>{metric(bm['tokens'], digits=0)}</td><td>{metric(cm['tokens'], digits=0)}</td></tr>
    <tr><td>Summary relevance</td><td>{metric(bm['judged'], '/5')} ({bm['judged_count']} judged)</td><td>{metric(cm['judged'], '/5')} ({cm['judged_count']} judged)</td></tr>'''
    release_label = 'REVIEW REQUIRED' if regressions or errors else 'NO NEW CASE REGRESSIONS DETECTED'
    release_reason = (f'{len(regressions)} case regression(s) and {len(errors)} error transition(s) detected. Overall accuracy alone does not establish release safety.'
                      if regressions or errors else 'No PASS-to-FAIL changes detected in this dataset. This does not guarantee production performance.')
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Regression Detection | Evaluation Evidence</title><style>
:root{--bg:#0b1020;--surface:#151d32;--surface2:#1d2942;--text:#eef2ff;--muted:#a6b4cf;--accent:#91a5ff;--good:#34d399;--bad:#fb7185;--warn:#fbbf24;--border:#2a3651}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 Inter,Segoe UI,Arial,sans-serif}.container{max-width:1280px;margin:auto;padding:34px 24px 70px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:26px}h1{font-size:30px;line-height:1.2;margin:6px 0}h2{font-size:19px;margin:0 0 16px}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:2px;font-size:11px}.muted,.sub{color:var(--muted)}.sub{display:block;font-size:12px;margin-top:3px}.small{font-size:12px}.pill,.tag{border-radius:99px;padding:6px 11px;background:var(--surface2);font-size:12px;display:inline-block}.pill{color:var(--accent)}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card,.panel{background:var(--surface);border:1px solid var(--border);border-radius:15px;padding:22px}.card .label{font-size:12px;color:var(--muted)}.card .value{font-size:29px;font-weight:800;margin:7px 0 3px}.panel{margin-top:20px}.good{color:var(--good)!important}.bad{color:var(--bad)!important}.warn{color:var(--warn)!important}.tag.good{background:#123c34}.tag.bad{background:#44202d}.tag.warn{background:#44341d}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;min-width:740px}th,td{text-align:left;border-bottom:1px solid var(--border);padding:12px 10px;vertical-align:top}th{font-size:12px;color:var(--muted)}tr:last-child td{border-bottom:0}.comparison-bars{min-width:120px}.bar{height:8px;border-radius:10px;margin:5px 0}.baseline{background:#6776a2}.candidate{background:var(--accent)}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:0 0 12px}.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}input,select,button{background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);padding:10px;font:inherit}input{flex:1;min-width:210px}button{cursor:pointer}details{min-width:130px}summary{cursor:pointer;color:var(--accent)}.evidence{padding:14px;margin-top:8px;background:var(--surface2);border-radius:10px;min-width:260px;max-width:500px;overflow-wrap:anywhere}.evidence p{margin:3px 0 13px}.decision{border-left:4px solid var(--warn)}.note{font-size:12px;color:var(--muted);margin-top:14px}footer{text-align:center;color:var(--muted);font-size:12px;margin-top:35px}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:500px){.grid{grid-template-columns:1fr}.container{padding:20px 12px}h1{font-size:25px}}@media print{body{background:white;color:black}.card,.panel{background:white;color:black;border-color:#ccc}.controls{display:none}.sub,.muted{color:#555}}
</style></head><body><div class="container"><header><div><div class="eyebrow">AI QUALITY ENGINEERING · EVALUATION EVIDENCE</div><h1>Model Regression Detection</h1><div class="muted">Customer support email classification · Same golden dataset comparison</div></div><div><span class="pill">Baseline v1 → Candidate v2 · Dataset v3</span><div class="sub">Generated: __GENERATED__</div></div></header>
<section class="grid">__CARDS__</section><section class="panel decision"><h2>Release review · __DECISION__</h2><p>__REASON__</p><div class="sub">Recommendation based on observed case-level transitions, not a configured production release gate.</div></section>
<section class="panel"><h2>Accuracy by category</h2><div class="legend"><span><i class="dot" style="background:#6776a2"></i>Baseline</span><span><i class="dot" style="background:var(--accent)"></i>Candidate</span></div><div class="table-wrap"><table><thead><tr><th>Category</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Visual comparison</th></tr></thead><tbody>__CATEGORIES__</tbody></table></div></section>
<section class="panel"><h2>Operational metrics</h2><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>__METRICS__</tbody></table></div><div class="note">Latency is the mean of recorded per-case values. Token totals are displayed only if all cases have usage. Summary relevance is displayed only if scores exist in the loaded checkpoints; classifier results alone do not include judge scores.</div></section>
<section class="panel"><h2>Case-level regression explorer</h2><p class="muted">Search by test ID, email text or category. Expand a case to inspect the actual model outputs and request metrics.</p><div class="controls"><input id="search" type="search" placeholder="Search cases, e.g. TC023 or billing" aria-label="Search test cases"><select id="filter" aria-label="Filter by transition"><option value="all">All cases</option><option value="regression">Regressions</option><option value="improvement">Improvements</option><option value="unchanged">Unchanged</option><option value="error-transition">Error transitions</option></select><button type="button" onclick="window.print()">Print / Save as PDF</button></div><div class="sub" id="visibleCount"></div><div class="table-wrap"><table><thead><tr><th>Case</th><th>Expected</th><th>Baseline</th><th>Candidate</th><th>Transition</th><th>Evidence</th></tr></thead><tbody>__DETAILS__</tbody></table></div></section>
<section class="panel"><h2>Evidence and reproducibility</h2><p>Every figure on this page is computed from saved evaluation outputs. This HTML is a generated snapshot; regenerate it after a new comparison run.</p><div class="small"><p><strong>Baseline:</strong> <code>__BASELINE__</code></p><p><strong>Candidate:</strong> <code>__CANDIDATE__</code></p><p><strong>Golden dataset:</strong> <code>__DATASET__</code></p><p><strong>Comparison:</strong> <code>__DIFF__</code></p></div><p class="note">Dataset provenance limitation: legacy checkpoints store fingerprints but not full dataset snapshots. Archive the exact dataset and prompt configuration used in each evaluation for stronger auditability. Results on 30 test cases do not establish production-wide accuracy.</p></section>
<footer>Model Regression Detection System · Offline report · No API requests</footer></div><script>
const search=document.getElementById('search'),filter=document.getElementById('filter'),rows=[...document.querySelectorAll('.case-row')];function applyFilters(){const q=search.value.trim().toLowerCase(),f=filter.value;let shown=0;for(const row of rows){const visible=(f==='all'||row.dataset.filter===f)&&row.dataset.search.includes(q);row.hidden=!visible;if(visible)shown++}document.getElementById('visibleCount').textContent=shown+' of '+rows.length+' cases shown'}search.addEventListener('input',applyFilters);filter.addEventListener('change',applyFilters);applyFilters();
</script></body></html>'''
    replacements = {
        '__GENERATED__': esc(generated), '__CARDS__': cards, '__DECISION__': esc(release_label),
        '__REASON__': esc(release_reason), '__CATEGORIES__': ''.join(category_rows),
        '__METRICS__': metric_rows, '__DETAILS__': ''.join(detail_rows),
        '__BASELINE__': esc(baseline_path), '__CANDIDATE__': esc(candidate_path),
        '__DATASET__': esc(dataset_path), '__DIFF__': esc(diff_path),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    print('Dashboard generated:', output.resolve())
    print(f"Baseline: {bm['accuracy']:.2f}% | Candidate: {cm['accuracy']:.2f}% | Regressions: {len(regressions)} | Improvements: {len(improvements)}")


def main():
    parser = argparse.ArgumentParser(description='Generate a static regression evidence dashboard')
    parser.add_argument('--baseline', default=DEFAULT_BASELINE)
    parser.add_argument('--candidate', default=DEFAULT_CANDIDATE)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--diff', default=DEFAULT_DIFF)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    make_dashboard(args.baseline, args.candidate, args.dataset, args.diff, args.output)


if __name__ == '__main__':
    main()
