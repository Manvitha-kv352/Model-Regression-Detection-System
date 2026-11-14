"""Inject historical score trend and metadata into an existing generated dashboard.
Run AFTER html_report.py and monitoring.py. No APIs or dependencies.
"""
import argparse
import html
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--html', default='reports/evaluation_dashboard.html')
    p.add_argument('--history', default='evaluations/run_history.json')
    p.add_argument('--monitor', default='evaluations/monitoring_report.json')
    p.add_argument('--last', type=int, default=12)
    args = p.parse_args()
    if args.last < 1:
        raise ValueError('--last must be positive')
    page_path = Path(args.html)
    page = page_path.read_text(encoding='utf-8')
    history = json.loads(Path(args.history).read_text(encoding='utf-8'))
    monitor = json.loads(Path(args.monitor).read_text(encoding='utf-8'))
    rows = history[-args.last:]
    if not rows:
        raise ValueError('No recorded runs')
    esc = lambda s: html.escape(str(s), quote=True)
    # Native SVG chart: works offline; no third-party JS and no fake scores.
    width, height = 900, 220
    left, top, right, bottom = 55, 15, 25, 35
    inner_w, inner_h = width-left-right, height-top-bottom
    points = []
    for i, run in enumerate(rows):
        x = left + i * inner_w / max(len(rows)-1, 1)
        y = top + (100-float(run['accuracy_percent'])) * inner_h/100
        points.append((x,y))
    grid = ''.join(f'<line x1="{left}" y1="{top+(100-t)*inner_h/100:.1f}" x2="{width-right}" y2="{top+(100-t)*inner_h/100:.1f}" stroke="#34405b"/><text x="4" y="{top+(100-t)*inner_h/100+4:.1f}" fill="#a6b4cf" font-size="12">{t}%</text>' for t in (0,25,50,75,100))
    line = ' '.join(f'{x:.1f},{y:.1f}' for x,y in points)
    dots = ''.join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#91a5ff"><title>{esc(rows[i]["prompt_version"])}: {rows[i]["accuracy_percent"]:.2f}%</title></circle>' for i,(x,y) in enumerate(points))
    svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Accuracy trend for last {len(rows)} recorded runs" style="width:100%;height:auto">{grid}<polyline points="{line}" fill="none" stroke="#91a5ff" stroke-width="3"/>{dots}</svg>'
    table = ''.join(f'<tr><td>{esc(r["prompt_version"])}</td><td>{esc(r["model"])}</td><td>{esc(r.get("timestamp_utc") or "Not recorded")}</td><td>{r["accuracy_percent"]:.2f}%</td></tr>' for r in rows)
    drift = monitor['drift']
    drift_text = f'{drift["moving_average"]:.2f}%' if drift.get('moving_average') is not None else 'Not enough comparable runs'
    panel = f'''<section class="panel" id="run-history"><h2>Evaluation history · last {len(rows)} runs</h2>
    <p class="muted">Recorded checkpoint history, not simulated data. Original timestamps may be unavailable for imported runs.</p>
    {svg}<p><strong>Drift status:</strong> {esc(drift['status'])} · <strong>7-run moving average:</strong> {esc(drift_text)} · <strong>Threshold:</strong> {esc(drift['threshold'])}%</p>
    <div class="table-wrap"><table><thead><tr><th>Prompt</th><th>Model</th><th>Run timestamp (UTC)</th><th>Accuracy</th></tr></thead><tbody>{table}</tbody></table></div></section>'''
    marker = '<section class="panel"><h2>Evidence and reproducibility</h2>'
    start, end = '<!-- HISTORY_PANEL_START -->', '<!-- HISTORY_PANEL_END -->'
    if start in page and end in page:
        a = page.index(start)
        b = page.index(end, a) + len(end)
        page = page[:a] + page[b:]
    if marker not in page:
        raise ValueError('Expected evidence section not found; check dashboard version')
    # Add true side-by-side summaries to the existing expandable regression evidence.
    page = page.replace('<strong>Baseline summary</strong><p>', '<div class="side-by-side"><div><strong>Baseline output</strong><p>')
    page = page.replace('</p>\n          <strong>Candidate summary</strong><p>', '</p></div><div><strong>Candidate output</strong><p>')
    page = page.replace('</p>\n          <strong>Baseline latency / tokens</strong>', '</p></div></div>\n          <strong>Baseline latency / tokens</strong>')
    page = page.replace('</style></head>', '.side-by-side{display:grid;grid-template-columns:1fr 1fr;gap:12px}.side-by-side>div{background:#151d32;padding:12px;border-radius:9px}@media(max-width:600px){.side-by-side{grid-template-columns:1fr}}\n</style></head>')
    page = page.replace(marker, start + panel + end + marker, 1)
    page_path.write_text(page, encoding='utf-8')
    print('Added run history and drift panel:', page_path.resolve())


if __name__ == '__main__':
    main()
