"""B5: sparse-gate L1 weight lambda vs search cost on the C3 full-edit data (runs/repair: every candidate edit run).
Same gate as repair_eval.train_gate except lambda; same component credit (max gate prob over the steps it produced)."""
import json, sys
import numpy as np, torch
import lch, repair_eval as RE, run_eval as E
R = E.R; D = f'{R}/runs/repair'


def train_gate(lam, seed=0):
    torch.manual_seed(seed)
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
        loss = torch.nn.functional.binary_cross_entropy(p, Y) + lam * g.mean() + 1e-4 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g = torch.sigmoid(X @ w + b)
    return dict(w=w.detach(), b=b.detach(), mu=mu, sd=sd, sparsity=float((g > 0.5).float().mean()))


B = {json.loads(l)['id']: json.loads(l) for l in open(f'{D}/base.jsonl')}
O = json.load(open(f'{D}/orders.json')); ok = {}
for l in open(f'{D}/edits.jsonl'):
    x = json.loads(l); ok[(x['id'], json.dumps(x['edit'], sort_keys=True))] = x['success']
K = lambda i, e: ok.get((int(i), json.dumps(e, sort_keys=True)), False)
fix = [i for i, o in O.items() if any(K(i, e) for e in o['gate'])]
env = E.ENVS['sciworld'](); tasks = {i: env.reset(int(i)).split('\n')[0] for i in fix}; env.close()
H = {i: torch.tensor(lch.encode(RE.step_texts(B[int(i)], tasks[i]), (20,), bs=8)[20]) for i in fix}
print('| lambda | gates>0.5 (train steps) | tries to 1st repair | repaired@B=1 | @B=2 | @B=4 | AUC |\n|---|---|---|---|---|---|---|')
for lam in (0.0, 0.001, 0.01, 0.1, 1.0):
    gt = train_gate(lam); tr, b1, b2, b4, au = [], 0, 0, 0, []
    for i in fix:
        r = B[int(i)]; p = torch.sigmoid(((H[i] - gt['mu']) / gt['sd']) @ gt['w'] + gt['b']).numpy()
        prov = r['prov'] or []; acts = [a for a, _ in r['trace']]

        def comp(e):
            if 'skip_plan' in e:
                v = [p[k] for k, pk in enumerate(prov) if pk == e['skip_plan']]
            elif 'drop_sub' in e:
                y = (r['subs'] or {}).get(e['drop_sub'], e['drop_sub']); v = [p[k] for k, a in enumerate(acts) if y in a]
            else:
                return float(np.mean(p)) if len(p) else 0.0
            return float(max(v)) if v else 0.0
        seq = sorted(O[i]['gate'], key=lambda e: -comp(e)); s = [K(i, e) for e in seq]
        t = s.index(True) + 1; tr.append(t); b1 += t <= 1; b2 += t <= 2; b4 += t <= 4
        pos = [k for k, x in enumerate(s) if x]; neg = [k for k, x in enumerate(s) if not x]
        if pos and neg:
            au.append(np.mean([a < c for a in pos for c in neg]))
    print(f'| {lam} | {gt["sparsity"]:.3f} | {np.mean(tr):.2f} | {b1} | {b2} | {b4} | {np.mean(au):.3f} |', flush=True)
