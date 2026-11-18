import json
from math import comb
D = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/runs/repair4'
k = lambda e: json.dumps(e, sort_keys=True)
ok = {}
for l in open(f'{D}/edits.jsonl'):
    x = json.loads(l); ok[(x['id'], k(x['edit']))] = x['success']
O = json.load(open(f'{D}/orders.json'))
hit = {m: {int(i) for i, o in O.items() if any(ok.get((int(i), k(e))) for e in o[m][:2])} for m in ('gate', 'random', 'simple')}
sign = lambda a, b: min(1.0, 2 * sum(comb(a + b, j) for j in range(min(a, b) + 1)) / 2 ** (a + b))
for a, b in [('gate', 'random'), ('gate', 'simple'), ('simple', 'random')]:
    w, l = len(hit[a] - hit[b]), len(hit[b] - hit[a]); print(f'TRAIN repairs {a} {len(hit[a])} vs {b} {len(hit[b])}: +{w}/-{l} p={sign(w, l):.3f}')
for m in ('gate', 'random', 'simple'):
    L = json.load(open(f'{D}/lib_{m}.json')); print(m, 'lib demos', len(L), 'with ops', sum('ops' in v for v in L.values()), 'exclude', sum(v.get('exclude', False) for v in L.values()))
