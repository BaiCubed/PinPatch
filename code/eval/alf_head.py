"""P1-2 on ALFWorld: learned search order (pre-registered PROGRESS.md 2026-11-18). TRAIN (train_lab) only, no LLM.

  alf_head.py data                    -> runs/p12a/lab.jsonl: for each lab game, from the initial state go to (and open)
                                         every receptacle once; label = the target object class is visible there
  alf_head.py train hidden|tfidf OUT  -> head + report (AUC; first-choice hit rate on held-out games vs the count prior)
  alf_head.py serve OUT PORT          -> HTTP /score {texts} -> {p, tau}   (used via ALF_SCORER_URL by meta_alf)
Held-out = 20% of lab games (seed 20261118)."""
import json, os, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import run_eval as E

R = E.R; OUT = f'{R}/runs/p12a'; os.makedirs(OUT, exist_ok=True); DATA = f'{OUT}/lab.jsonl'
NUM = lambda r: int(re.search(r'(\d+)$', r).group(1)) if re.search(r'\d+$', r) else 0


def one(i):
    env = E.ENVS['alfworld'](); obs = env.reset(i)
    goal = re.search(r'Your task is to: (.+?)\.?(?:\n|$)', obs).group(1).lower()
    obj = next((m.group(1) for rx, _ in E.ALF_GOALS for m in [re.search(rx, goal)] if m), None)
    if obj is None:
        return []
    rows = []

    def st(act):  # -> observation, or None when the server's step budget for this episode is used up
        j = env.post('/step', action=act)
        if 'observation' not in j:
            return None
        env.available = j.get('available_actions', [])
        return j['observation']

    def visit(r):  # go to (and open) r; objects never move, so after the budget runs out reset the game and redo the pair
        for _ in range(2):
            o = st(f'go to {r}')
            if o is not None and f'open {r}' in env.available:
                o = st(f'open {r}')
            if o is not None:
                return o
            env.reset(i)
        raise RuntimeError(f'game {i}: cannot visit {r}')
    for r in [x[6:] for x in env.available if x.startswith('go to ')]:
        o = visit(r)
        rows.append(dict(id=i, goal=goal, obj=obj, recep=r, y=int(bool(re.search(r'\b' + re.escape(obj) + r' \d+', o)))))
    return rows


def data():  # resumable: skips games already in lab.jsonl; LAB_LIMIT = only the first N lab games (server restarts between chunks)
    ids = E.ids('alfworld', 'train_lab')[: int(os.environ.get('LAB_LIMIT', 10 ** 9))]
    done = {json.loads(l)['id'] for l in open(DATA)} if os.path.exists(DATA) else set()
    ids = [i for i in ids if i not in done]
    with open(DATA, 'a') as f, ThreadPoolExecutor(8) as ex:
        for k, rows in enumerate(ex.map(one, ids)):
            for x in rows:
                f.write(json.dumps(x) + '\n')
    print('LAB_DONE', DATA)


def first_hit(X, p):
    by = defaultdict(list)
    for x, q in zip(X, p):
        by[x['id']].append((-q, NUM(x['recep']), x['recep'], x['y']))
    return float(np.mean([sorted(v)[0][3] for v in by.values()]))


def train(kind, out):
    import torch, lch, lch_text
    X = [json.loads(l) for l in open(DATA)]
    for x in X:
        x['text'] = E.alf_head_text(x['goal'], x['obj'], x['recep'])
    y = np.array([float(x['y']) for x in X])
    g = sorted({x['id'] for x in X}); random.Random(20261118).shuffle(g); hold = set(g[: len(g) // 5])
    te = np.array([x['id'] in hold for x in X]); tr = ~te; texts = [x['text'] for x in X]
    Xte = [x for x, m in zip(X, te) if m]
    pri = E.alf_prior()['where']
    prior_p = np.array([pri.get(x['obj'], {}).get(re.sub(r' \d+$', '', x['recep']), 0) for x in Xte], dtype=float)
    if kind == 'tfidf':
        h = lch_text.fit_tfidf([t for t, m in zip(texts, tr) if m], y[tr]); p = lch_text.predict(h, [t for t, m in zip(texts, te) if m])
        rep = dict(layer=None); full = lambda: lch_text.fit_tfidf(texts, y)
    else:
        H = lch.encode(texts, (14, 20, 27)); rep = {}; best = None
        for L, M in H.items():
            hh = lch.fit(M[tr], y[tr]); pp = lch.prob(hh, M[te]); a = lch_text.auc(pp, y[te]); rep[f'h{L}'] = round(a, 3)
            if best is None or a > best[1]:
                best = (L, a, pp)
        L, _, p = best; rep['layer'] = L
        full = lambda: dict(lch.fit(H[L], y), layer=L)
    rep.update(kind=kind, auc=round(lch_text.auc(p, y[te]), 3), first_hit=round(first_hit(Xte, p), 4),
               prior_auc=round(lch_text.auc(prior_p, y[te]), 3), prior_first_hit=round(first_hit(Xte, prior_p), 4),
               n=len(X), n_pos=int(y.sum()), games=len(g))
    h = full(); h['tau'] = 0.5; h['report'] = rep; h['kind'] = kind
    torch.save(h, out); print(json.dumps(rep), '->', out)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'data':
        data()
    elif c == 'train':
        train(sys.argv[2], sys.argv[3])
    else:
        import ws_head  # same service: /score with lch/lch_text heads
        ws_head.serve(sys.argv[2], sys.argv[3])
