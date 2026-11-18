"""C2/E3: compare credit orderings for repairing failed procedures (ScienceWorld TRAIN).
For each failed base episode with at least one repairing edit, an ordering ranks the candidate edits; the score is
the number of tries until the first repair. Step credit -> component credit: a plan-step edit gets the max credit of
the steps that plan step produced (provenance); a slot edit gets the max credit of the steps whose action mentions
the slot's filler; a demo edit gets the mean credit of all steps.

  repair_eval.py BASE.jsonl EDITS.jsonl      (GPU: step features for the gate)
Orderings: random (exact expectation), recency (latest plan step first), lch (learned violation head), gate (sparse
outcome gate trained on episode success only, runs/lch/v1 data), oracle (lower bound)."""
import json, sys

import numpy as np
import torch

import lch
import run_eval as E

R = E.R


def train_gate():
    """Sparse outcome gate on TRAIN no-shield episodes (same data as credit.py): noisy-or over step gates, L1."""
    rows = [json.loads(l) for l in open(f'{R}/runs/lch/v1.jsonl')]
    X = torch.tensor(np.load(f'{R}/runs/lch/v1.npz')['h20']); mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    succ = {}
    for h in 'abcd':
        for l in open(f'{R}/runs/lch1_sci_noshield_train_{h}/episodes.jsonl'):
            e = json.loads(l); succ[e['id']] = e['success']
    eps = sorted({r['ep'] for r in rows}); ix = {e: i for i, e in enumerate(eps)}
    Ei = torch.tensor([ix[r['ep']] for r in rows]); Y = torch.tensor([0.0 if succ[e] else 1.0 for e in eps])
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.full((1,), -4.0, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-3)
    for _ in range(600):
        g = torch.sigmoid(X @ w + b)
        ls = torch.zeros(len(eps)).index_add(0, Ei, torch.log1p(-g.clamp(max=1 - 1e-6)))
        p = (1 - torch.exp(ls)).clamp(1e-6, 1 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(p, Y) + 0.01 * g.mean() + 1e-4 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return dict(w=w.detach(), b=b.detach(), mu=mu, sd=sd)


def step_texts(r, task):
    return [E.lch_text(task, r['trace'][:k], a) for k, (a, _) in enumerate(r['trace'])]


def main(base_path, edits_path):
    B = {json.loads(l)['id']: json.loads(l) for l in open(base_path)}
    ED = {}
    for l in open(edits_path):
        x = json.loads(l); ED.setdefault(x['id'], []).append(x)
    env = E.ENVS['sciworld'](); tasks = {i: env.reset(i).split('\n')[0] for i in ED}; env.close()
    head = torch.load(f'{R}/runs/lch/v1.pt', weights_only=False); gate = train_gate()
    res = {k: [] for k in ('random', 'recency', 'lch', 'gate', 'oracle')}; n_fail = len(ED); n_rep = 0; ms = []
    orders = {}
    for i, eds in ED.items():
        fix = [e for e in eds if e['success']]
        r = B[i]; m = len(eds)
        H = lch.encode(step_texts(r, tasks[i]), (20,), bs=8)[20]
        Xl = torch.tensor(H if head['layer'] == 20 else lch.encode(step_texts(r, tasks[i]), (head['layer'],))[head['layer']])
        p_lch = torch.sigmoid(((Xl - head['mu']) / head['sd']) @ head['w'] + head['b']).numpy()
        p_gate = torch.sigmoid(((torch.tensor(H) - gate['mu']) / gate['sd']) @ gate['w'] + gate['b']).numpy()
        prov = r['prov'] or []; acts = [a for a, _ in r['trace']]

        def comp(e, p):
            if 'skip_plan' in e:
                v = [p[k] for k, pk in enumerate(prov) if pk == e['skip_plan']]
            elif 'drop_sub' in e:
                y = (r['subs'] or {}).get(e['drop_sub'], e['drop_sub'])
                v = [p[k] for k, a in enumerate(acts) if y in a]
            else:
                return float(np.mean(p)) if len(p) else 0.0
            return float(max(v)) if v else 0.0

        def rank(order):
            return next(k for k, e in enumerate(order) if e['success']) + 1
        import random as _r
        orders[i] = dict(gate=[e['edit'] for e in sorted(eds, key=lambda e: -comp(e['edit'], p_gate))],
                         lch=[e['edit'] for e in sorted(eds, key=lambda e: -comp(e['edit'], p_lch))],
                         recency=[e['edit'] for e in sorted(eds, key=lambda e: -e['edit'].get('skip_plan', -1))],
                         random=[e['edit'] for e in _r.Random(i).sample(eds, len(eds))])
        if not fix:
            continue
        n_rep += 1
        res['random'].append((m + 1) / (len(fix) + 1)); ms.append((m, len(fix)))
        res['recency'].append(rank(sorted(eds, key=lambda e: -e['edit'].get('skip_plan', -1))))
        res['lch'].append(rank(sorted(eds, key=lambda e: -comp(e['edit'], p_lch))))
        res['gate'].append(rank(sorted(eds, key=lambda e: -comp(e['edit'], p_gate))))
        res['oracle'].append(1)
    from math import comb
    print(f'failed episodes {n_fail}; repairable by one edit {n_rep}')
    p1 = np.mean([s / m for m, s in ms]); p3 = np.mean([1 - comb(m - s, min(3, m)) / comb(m, min(3, m)) for m, s in ms])
    print(f'  random   mean tries {np.mean(res["random"]):.2f}   first-try repair {p1:.3f}   within 3 {p3:.3f}  (exact expectation)')
    for k in ('recency', 'lch', 'gate', 'oracle'):
        v = np.array(res[k])
        print(f'  {k:8s} mean tries {v.mean():.2f}   first-try repair {np.mean(v <= 1):.3f}   within 3 {np.mean(v <= 3):.3f}')
    g = np.array(res['gate'])
    for k in ('random', 'recency', 'lch'):
        o = np.array(res[k]); w = int((g < o).sum()); l = int((g > o).sum()); n = w + l
        p = min(1.0, 2 * sum(comb(n, j) for j in range(min(w, l) + 1)) / 2 ** n) if n else 1.0
        print(f'  gate vs {k:7s}: fewer tries in {w}, more in {l}, sign test p={p:.3g}')
    json.dump(dict(res={k: list(map(float, v)) for k, v in res.items()}, ms=ms), open(f'{R}/runs/repair/orderings.json', 'w'))
    json.dump({str(k): v for k, v in orders.items()}, open(f'{R}/runs/repair/orders.json', 'w'))
    kinds = {}
    for eds in ED.values():
        for e in eds:
            if e['success']:
                kinds[next(iter(e['edit']))] = kinds.get(next(iter(e['edit'])), 0) + 1
    print('repairing edits by kind', kinds)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
