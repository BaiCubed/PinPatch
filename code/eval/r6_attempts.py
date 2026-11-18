"""R6/R2a support: per-failure rank of the first repairing edit under each candidate order (C3 and C4 spaces).
Oracle = outcome oracle (a repairing edit first whenever one exists): the upper bound for any credit signal.
random = exact expectation (m+1)/(r+1) for r repairing among m candidates. -> runs/paper_data/attempts.csv"""
import csv, json
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/runs'
rows = []
for space, d in (('c3', 'repair'), ('c4', 'repair4')):
    O = json.load(open(f'{R}/{d}/orders.json')); ok = {}
    for l in open(f'{R}/{d}/edits.jsonl'):
        x = json.loads(l); ok[(int(x['id']), json.dumps(x['edit'], sort_keys=True))] = x['success']
    K = lambda i, e: ok.get((int(i), json.dumps(e, sort_keys=True)))
    for m in sorted(next(iter(O.values())).keys()):
        if m == 'random' and space == 'c3':
            continue
        firsts, unk, n_rep = [], 0, 0
        for i, o in O.items():
            cand = o[m]; known = [K(i, e) for e in cand]
            if None in known:
                unk += 1; continue
            if any(known):
                n_rep += 1; firsts.append(known.index(True) + 1)
        rows.append(dict(space=space, method=m, n_fail=len(O), n_repairable=n_rep, n_unknown=unk,
                         mean_first=round(sum(firsts) / max(1, len(firsts)), 2), median_first=sorted(firsts)[len(firsts) // 2] if firsts else ''))
    # exact random expectation and oracle, over failures whose whole candidate list was run
    base = next(iter(O.values())); m0 = 'gate'
    ex, orc, n = 0.0, 0, 0
    for i, o in O.items():
        known = [K(i, e) for e in o[m0]]
        if None in known or not any(known):
            continue
        r_, mm = sum(known), len(known); ex += (mm + 1) / (r_ + 1); n += 1
    rows.append(dict(space=space, method='random_exact', n_fail=len(O), n_repairable=n, n_unknown='', mean_first=round(ex / max(1, n), 2), median_first=''))
    rows.append(dict(space=space, method='oracle', n_fail=len(O), n_repairable=n, n_unknown='', mean_first=1.0, median_first=1))
with open(f'{R}/paper_data/attempts.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
