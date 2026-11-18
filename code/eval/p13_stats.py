"""P1-3 held-out task types: hold same type (near) / same family (far) out of the demo pool.
-> runs/paper_data/heldout_types.csv (overall + per type) ; paired vs full pool (frozen_c3r2/test_gate)."""
import csv, glob, json
import export_paper_data as X
g = json.load(open(f'{X.R}/data/splits/sciworld_games.json'))
rd = sorted(glob.glob(f'{X.RUNS}/*react_demo*test*/episodes.jsonl'))
cols = dict(full='frozen_c3r2/test_gate', hold_type='p13_hot/test_hold_type', hold_family='p13_hot/test_hold_family',
            react='r4_sci_react_test_serial')
T = {k: X.jl(f'{X.RUNS}/{v}/episodes.jsonl') for k, v in cols.items()}
if rd: T['react_demo'] = X.jl(rd[0]); print('react_demo from', rd[0])
F = T['full']; rows = []
def frac(D, ids):
    s = sum(D[i].get('steps', 0) or 0 for i in ids); l = sum(D[i].get('llm_steps', 0) or 0 for i in ids) if 'llm_steps' in next(iter(D.values())) else None
    return round(l / s, 3) if (l is not None and s) else ''
for k, D in T.items():
    ids = set(F) & set(D); w = sum(bool(D[i]['success']) > bool(F[i]['success']) for i in ids); l = sum(bool(D[i]['success']) < bool(F[i]['success']) for i in ids)
    rows.append(dict(level='overall', task_type='all', family='all', n=len(D), config=k, success=sum(bool(r['success']) for r in D.values()),
                     vs_full_better=w if k != 'full' else '', vs_full_worse=l if k != 'full' else '', sign_p=f'{X.sign(w, l):.3g}' if k != 'full' else '',
                     llm_step_frac=frac(D, list(D)), mean_score=X.mean([float(r.get('score', 0) or 0) for r in D.values()])))
by = {}
for i in F: by.setdefault(g[str(i)], []).append(i)
for t, ids in sorted(by.items(), key=lambda x: -len(x[1])):
    for k, D in T.items():
        rows.append(dict(level='type', task_type=t, family=t.split('-')[0], n=len(ids), config=k, success=sum(bool(D.get(i, {}).get('success', False)) for i in ids),
                         vs_full_better='', vs_full_worse='', sign_p='', llm_step_frac=frac(D, [i for i in ids if i in D]) if all(i in D for i in ids) else '',
                         mean_score=''))
with open(f'{X.OUT}/heldout_types.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    if r['level'] == 'overall': print(r)
print('per type (full / type / family / react):')
for t in by:
    print(f"{t:40s}", [r['success'] for r in rows if r['task_type'] == t])
