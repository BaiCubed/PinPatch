import json, os
from math import comb
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/runs'
def load(p):
    d = {}
    for s in range(8):
        f = f'{R}/{p}_s{s}/episodes.jsonl'
        if os.path.exists(f):
            for l in open(f):
                try: e = json.loads(l); d[e['id']] = e
                except Exception: pass
    return d
def sign(a, b):
    n = a + b; k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
M = {k: load(k) for k in ['ho2_base', 'ho2_gate', 'ho2_random', 'ho2_gater2', 'ho4_gate', 'ho4_random', 'ho4_simple']}
for k, d in M.items():
    print(f'{k:12s} n={len(d)} succ={sum(e["success"] for e in d.values())} mean={sum(e["score"] for e in d.values())/max(1,len(d)):.1f}')
def cmp(a, b):
    ids = set(M[a]) & set(M[b]); w = sum(M[a][i]['success'] and not M[b][i]['success'] for i in ids); l = sum(M[b][i]['success'] and not M[a][i]['success'] for i in ids)
    print(f'{a} vs {b}: +{w}/-{l} p={sign(w, l):.3f} (n={len(ids)})')
for a, b in [('ho4_gate', 'ho4_random'), ('ho4_gate', 'ho4_simple'), ('ho4_gate', 'ho2_base'), ('ho4_random', 'ho2_base'), ('ho4_simple', 'ho2_base'),
             ('ho4_gate', 'ho2_gate'), ('ho4_random', 'ho2_random'), ('ho4_gate', 'ho2_gater2')]:
    cmp(a, b)
