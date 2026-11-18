import json, os, sys
from math import comb
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/runs'
def load(r):
    d = {}
    for l in open(f'{R}/{r}/episodes.jsonl'):
        try: e = json.loads(l); d.setdefault(e['id'], e)
        except Exception: pass
    return d
def sign(w, l):
    n = w + l; return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
REF = dict(full='frozen_c3r2/test_gate', lch='frozen_c3/test_lch', r1='frozen_c3/test_evolved', v7='frozen_sci/test_meta_replay_v7',
           ho_full='ho2_gater2', ho_base='ho2_base')
def get(k):
    if k == 'ho_full' or k == 'ho_base':
        d = {}
        for s in range(8): d.update(load(f'{REF[k]}_s{s}'))
        return d
    return load(REF.get(k, k))
pairs = [x.split(':') for x in sys.argv[1:]]
for a, b in pairs:
    A, B = get(a), get(b); ids = set(A) & set(B)
    w = sum(A[i]['success'] and not B[i]['success'] for i in ids); l = sum(B[i]['success'] and not A[i]['success'] for i in ids)
    sw = sum(A[i]["score"] > B[i]["score"] for i in ids); sl = sum(A[i]["score"] < B[i]["score"] for i in ids)
    na = sum(A[i]["score"] < 0 for i in ids); nb = sum(B[i]["score"] < 0 for i in ids)
    ma = sum(A[i]['score'] for i in ids) / len(ids); mb = sum(B[i]['score'] for i in ids) / len(ids)
    print(f'{a.split("/")[-1]} vs {b}: n={len(ids)} succ {sum(A[i]["success"] for i in ids)} vs {sum(B[i]["success"] for i in ids)} '
          f"+{w}/-{l} p={sign(w, l):.3f} | mean {ma:.1f} vs {mb:.1f} | score +{sw}/-{sl} p={sign(sw, sl):.3g} | neg-score eps {na} vs {nb}")
