"""P0-1: learned constraint head v2 (TRAIN interaction only; no hand-written rule is used to make labels).
  lch2.py build NEW_RUN_DIR...        v2 data = v1 examples + examples from new no-shield episodes -> runs/lch/v2.{jsonl,npz}
  lch2.py credit                      sparse outcome gate on v2 data -> credit labels (failed episode, g_t > 0.5) in v2c.npy
  lch2.py train VARIANT OUT.pt        a: v2 data, layer 20 | b: + credit labels | c: b with layers 14+20 | d: c + per-type tau
  lch2.py serve HEAD.pt PORT          /score like lch.py serve; per-type heads return a monotone score with tau 0.5
Held-out for thresholds: 20% of episodes (seeded), threshold = argmax recall - 3 * false-block (as v1)."""
import json, os, random, sys

import numpy as np

import lch
import run_eval as E

R = E.R; D = f'{R}/runs/lch'
V1_DIRS = [f'{R}/runs/lch1_sci_noshield_train_{h}' for h in 'abcd']


def episodes(dirs):
    out = {}
    for d in dirs:
        for l in open(f'{d}/episodes.jsonl'):
            try:
                r = json.loads(l); out.setdefault(r['id'], r)
            except json.JSONDecodeError:
                pass
    return out


def build(new_dirs):
    lch.data(new_dirs, f'{D}/v2new.jsonl')
    lch.feats(f'{D}/v2new.jsonl', f'{D}/v2new.npz', (14, 20))
    a, b = np.load(f'{D}/v1.npz'), np.load(f'{D}/v2new.npz')
    rows = [json.loads(l) for l in open(f'{D}/v1.jsonl')] + [json.loads(l) for l in open(f'{D}/v2new.jsonl')]
    with open(f'{D}/v2.jsonl', 'w') as f:
        for x in rows:
            f.write(json.dumps(x) + '\n')
    np.savez(f'{D}/v2.npz', y=np.concatenate([a['y'], b['y']]), ep=np.concatenate([a['ep'], b['ep']]),
             h14=np.concatenate([a['h14'], b['h14']]), h20=np.concatenate([a['h20'], b['h20']]))
    json.dump(new_dirs, open(f'{D}/v2_dirs.json', 'w'))
    print('v2 data', len(rows), 'examples,', int(a['y'].sum() + b['y'].sum()), 'violations')


def credit():
    import torch
    d = np.load(f'{D}/v2.npz'); ep = d['ep']
    succ = {i: r['success'] for i, r in episodes(V1_DIRS + json.load(open(f'{D}/v2_dirs.json'))).items()}
    X = torch.tensor(d['h20']); mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    eps = sorted(set(ep.tolist())); ix = {e: i for i, e in enumerate(eps)}
    Ei = torch.tensor([ix[e] for e in ep.tolist()]); Y = torch.tensor([0.0 if succ[e] else 1.0 for e in eps])
    torch.manual_seed(0); w = torch.zeros(X.shape[1], requires_grad=True); b = torch.full((1,), -4.0, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-3)
    for _ in range(600):  # identical objective to repair_eval.train_gate (noisy-or over steps, outcome only, L1 0.01)
        g = torch.sigmoid(X @ w + b)
        ls = torch.zeros(len(eps)).index_add(0, Ei, torch.log1p(-g.clamp(max=1 - 1e-6)))
        p = (1 - torch.exp(ls)).clamp(1e-6, 1 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(p, Y) + 0.01 * g.mean() + 1e-4 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g = torch.sigmoid(X @ w + b).numpy()
    failed = np.array([not succ[e] for e in ep.tolist()])
    yc = ((d['y'] == 1) | (failed & (g > 0.5))).astype(int)
    np.save(f'{D}/v2c.npy', yc)
    print('credit labels:', int(d['y'].sum()), 'outcome violations ->', int(yc.sum()), 'with credit-selected steps')


def pick_tau_strict(p, y, fb_max=0.002):
    """The deployed v1 rule (tau 0.9761): maximise recall subject to false-block <= 0.2% on the held-out episodes."""
    best = (-1.0, 1.0)
    for t in np.unique(np.round(p, 4)):
        fb = ((p > t) & (y == 0)).sum() / max(1, (y == 0).sum())
        if fb <= fb_max:
            best = max(best, (((p > t) & (y == 1)).sum() / max(1, (y == 1).sum()), -float(t)))
    return -best[1] if best[0] >= 0 else 1.0


def acttype(a):
    return 'focus' if a.startswith('focus on') else 'move' if a.startswith(('move ', 'pour ', 'put ')) else 'other'


def train(variant, out):
    import torch
    d = np.load(f'{D}/v2.npz'); ep = d['ep']; acts = [json.loads(l)['act'] for l in open(f'{D}/v2.jsonl')]
    y_eval = d['y']  # thresholds and the report always use the outcome labels (what the shield must stop)
    y_fit = np.load(f'{D}/v2c.npy') if variant in 'bcd' else d['y']
    if variant == 'e':  # drop positives whose action string also ends some successful TRAIN episode (answers, not violations)
        eps_ = episodes(V1_DIRS + json.load(open(f'{D}/v2_dirs.json')))
        final_ok = {r['trace'][-1][0] for r in eps_.values() if r['success'] and r.get('trace')}
        y_fit = np.array([0 if (yy == 1 and a_ in final_ok) else yy for yy, a_ in zip(d['y'], acts)])
        y_eval = y_fit
        print('e: positives', int(d['y'].sum()), '->', int(y_fit.sum()), flush=True)
    X = np.concatenate([d['h14'], d['h20']], 1) if variant in 'cd' else d['h20']
    layers = [14, 20] if variant in 'cd' else [20]
    eps = sorted(set(ep.tolist())); random.Random(20261107).shuffle(eps); hold = set(eps[: len(eps) // 5])
    te = np.array([e in hold for e in ep.tolist()])
    h = lch.fit(X[~te], y_fit[~te]); p = lch.prob(h, X[te]); yt = y_eval[te]
    tau = pick_tau_strict(p, yt); rep = dict(auc=round(lch.auc(p, yt), 4))
    rep['global'] = dict(tau=tau, recall=float(((p > tau) & (yt == 1)).sum() / max(1, yt.sum())),
                         false_block=float(((p > tau) & (yt == 0)).sum() / max(1, (yt == 0).sum())))
    tt = None
    if variant == 'd':
        ty = np.array([acttype(a) for a in acts])[te]; tt = {}
        for t in ('focus', 'move', 'other'):
            m = ty == t
            tt[t] = pick_tau_strict(p[m], yt[m]) if yt[m].sum() > 0 else 1.0
        blk = np.array([p[k] > tt[ty[k]] for k in range(len(p))])
        rep['per_type'] = dict(tau=tt, recall=float((blk & (yt == 1)).sum() / max(1, yt.sum())),
                               false_block=float((blk & (yt == 0)).sum() / max(1, (yt == 0).sum())))
    hf = lch.fit(X, y_fit)  # refit on all TRAIN examples; thresholds from the held-out split
    torch.save(dict(layers=layers, layer=layers[-1], tau=tau, tau_type=tt, variant=variant, **hf, report=rep,
                    n=int(len(y_fit)), n_pos=int(y_fit.sum())), out)
    print(variant, json.dumps(rep), '->', out)


def serve(head, port):
    import torch, uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    h = torch.load(head, weights_only=False); Ls = tuple(h.get('layers') or [h['layer']]); app = FastAPI()

    class Q(BaseModel):
        texts: list

    @app.post('/score')
    def score(q: Q):
        H = lch.encode(q.texts, Ls, bs=1); X = np.concatenate([H[L] for L in Ls], 1)
        p = lch.prob(h, X)
        if not h.get('tau_type'):
            return dict(p=[float(x) for x in p], tau=h['tau'])
        lg = lambda x: np.log(np.clip(x, 1e-9, 1 - 1e-9) / np.clip(1 - x, 1e-9, 1))
        out = []
        for t_, x in zip(q.texts, p):  # blocking iff p > tau_type  <=>  score > 0.5
            a = t_.rsplit('Candidate action: ', 1)[-1].strip()
            out.append(float(1 / (1 + np.exp(-(lg(x) - lg(h['tau_type'][acttype(a)]))))))
        return dict(p=out, tau=0.5)

    lch.model(); uvicorn.run(app, host='127.0.0.1', port=int(port))


def serve_multi(port, *pairs):
    """One encoder, several heads: NAME=HEAD.pt ... -> POST /NAME/score (set LCH_URL=http://127.0.0.1:PORT/NAME)."""
    import torch, uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    heads = {n: torch.load(f, weights_only=False) for n, f in (x.split('=', 1) for x in pairs)}
    Ls = tuple(sorted({L for h in heads.values() for L in (h.get('layers') or [h['layer']])})); app = FastAPI()

    class Q(BaseModel):
        texts: list
    lg = lambda x: np.log(np.clip(x, 1e-9, 1 - 1e-9) / np.clip(1 - x, 1e-9, 1))

    def make(name, h):
        @app.post(f'/{name}/score')
        def score(q: Q):
            H = lch.encode(q.texts, Ls, bs=1); X = np.concatenate([H[L] for L in (h.get('layers') or [h['layer']])], 1)
            p = lch.prob(h, X)
            if not h.get('tau_type'):
                return dict(p=[float(x) for x in p], tau=h['tau'])
            out = [float(1 / (1 + np.exp(-(lg(x) - lg(h['tau_type'][acttype(t_.rsplit('Candidate action: ', 1)[-1].strip())])))))
                   for t_, x in zip(q.texts, p)]
            return dict(p=out, tau=0.5)
    for n, h in heads.items():
        make(n, h)
    lch.model(); uvicorn.run(app, host='127.0.0.1', port=int(port))


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'build':
        build(sys.argv[2:])
    elif c == 'credit':
        credit()
    elif c == 'train':
        train(sys.argv[2], sys.argv[3])
    elif c == 'serve':
        serve(sys.argv[2], sys.argv[3])
    elif c == 'serve_multi':
        serve_multi(sys.argv[2], *sys.argv[3:])
