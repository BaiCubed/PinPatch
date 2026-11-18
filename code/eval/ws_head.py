"""P1-2: learned purchase constraint on WebShop (pre-registered PROGRESS.md 2026-11-18). TRAIN only.

Data: runs/ws_sim/train_lab_2000.jsonl (ws_sim.py on train_lab: top-10 results of two query forms, each bought in a
fresh episode with the goal's option values clicked; real reward). Label y = reward == 1. Input text = ws_head_text.
Heads, all selected on a 20% split of lab GOALS (seed 20261118), never on held-out/dev/test:
  hidden : Qwen2.5-7B last-token hidden state (layers 14 / 20 / 27; best held-out AUC), logistic regression (lch.fit)
  tfidf  : word 1-2 gram TF-IDF of the same text, logistic regression (lch_text.fit_tfidf)
Threshold tau (used by meta_ws to keep browsing instead of committing): the value that maximises the mean reward of the
simulated policy "buy the first item in search order whose p >= tau, else the max-p item" on the lab held-out goals.

  ws_head.py train hidden|tfidf OUT.pt     ws_head.py serve OUT.pt PORT   (HTTP /score {texts} -> {p, tau})"""
import json, random, sys
from collections import defaultdict

import numpy as np

import run_eval as E

R = E.R; DATA = f'{R}/runs/ws_sim/train_lab_2000.jsonl'


def rows():
    out = []
    for l in open(DATA):
        x = json.loads(l)
        x['text'] = E.ws_head_text(x['goal'], x['title'], x['price'], x['n_match'], x['n_opt'])
        out.append(x)
    return out


def split(X):
    g = sorted({x['id'] for x in X}); random.Random(20261118).shuffle(g); hold = set(g[: len(g) // 5])
    return np.array([x['id'] in hold for x in X])


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def policy_reward(X, p, tau):
    """Mean reward on held-out goals of: per (goal, query form desc_opts), go in rank order, buy the first item with
    p >= tau (at most 3 visits, the WebShop step budget), else the visited item with the highest p."""
    by = defaultdict(list)
    for x, q in zip(X, p):
        if x['q'] == 'desc_opts':
            by[x['id']].append((x['rank'], q, x['reward']))
    tot = []
    for v in by.values():
        v = sorted(v)[:3]
        pick = next((z for z in v if z[1] >= tau), max(v, key=lambda z: z[1]))
        tot.append(pick[2])
    return float(np.mean(tot)) if tot else 0.0


def pick_tau(X, p):
    best = max((policy_reward(X, p, t), t) for t in np.unique(np.round(np.r_[p, 1.01], 3)))
    return best[1], best[0]


def train(kind, out):
    import torch
    import lch, lch_text
    X = rows(); y = np.array([float(x['reward'] >= 1) for x in X]); te = split(X); tr = ~te
    texts = [x['text'] for x in X]
    if kind == 'tfidf':
        h = lch_text.fit_tfidf([t for t, m in zip(texts, tr) if m], y[tr])
        p = lch_text.predict(h, [t for t, m in zip(texts, te) if m]); rep = dict(layer=None)
        full = lambda: lch_text.fit_tfidf(texts, y)
    else:
        H = lch.encode(texts, (14, 20, 27)); rep = {}; best = None
        for L, M in H.items():
            hh = lch.fit(M[tr], y[tr]); pp = lch.prob(hh, M[te]); a = auc(pp, y[te]); rep[f'h{L}'] = round(a, 3)
            if best is None or a > best[1]:
                best = (L, a, pp)
        L, _, p = best; rep['layer'] = L
        full = lambda: dict(lch.fit(H[L], y), layer=L)
    Xte = [x for x, m in zip(X, te) if m]
    tau, pr = pick_tau(Xte, p)
    rep.update(kind=kind, auc=round(auc(p, y[te]), 3), tau=tau, ho_policy_reward=round(pr, 4), n=len(X), n_pos=int(y.sum()),
               rank0_reward=round(policy_reward(Xte, np.array([-x['rank'] for x in Xte], dtype=float), 1e9), 4))
    tau = float(tau); rep['tau'] = tau
    h = full(); h['tau'] = tau; h['report'] = rep; h['kind'] = kind
    torch.save(h, out); print(json.dumps(rep), '->', out)


def serve(head, port):
    import torch, uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    import __main__, lch, lch_text
    __main__.Tfidf = lch_text.Tfidf
    h = torch.load(head, weights_only=False); app = FastAPI()

    class Q(BaseModel):
        texts: list

    @app.post('/score')
    def score(q: Q):
        if h['kind'] == 'tfidf':
            p = lch_text.predict(h, q.texts)
        else:
            p = lch.prob(h, lch.encode(q.texts, (h['layer'],), bs=4)[h['layer']])
        return dict(p=[float(x) for x in p], tau=float(h['tau']))

    if h['kind'] != 'tfidf':
        lch.model()
    uvicorn.run(app, host='127.0.0.1', port=int(port))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(sys.argv[2], sys.argv[3])
    else:
        serve(sys.argv[2], sys.argv[3])
