"""Fit the WebShop item scorer on the TRAIN purchase simulation (ws_sim.py): least squares on title features,
one model before opening an item (pre) and one after (post, adds option matches). Prints held-out choice quality."""
import json, re, sys
import numpy as np
import run_eval as E

rows = [json.loads(l) for l in open(sys.argv[1])]


def feats(goal, title, price, rank, n_match=None, n_opt=None):
    f = E.ws_features(goal, title, price, rank)
    if n_match is not None:
        f += [n_match / max(1, n_opt), float(n_opt == 0 or n_match == n_opt)]
    return f


def fit(post):
    X = np.array([feats(r['goal'], r['title'], r['price'], r['rank'], *( (r['n_match'], r['n_opt']) if post else ())) for r in rows])
    y = np.array([r['reward'] for r in rows])
    ids = np.array([r['id'] for r in rows]); qs = np.array([r['q'] for r in rows])
    X1 = np.c_[X, np.ones(len(X))]
    # 2-fold by goal id: choice quality = mean reward of the argmax item per (goal, query)
    u = np.unique(ids); half = set(u[: len(u) // 2])
    m = np.array([i in half for i in ids]); res = []
    for tr, te in ((m, ~m), (~m, m)):
        w = np.linalg.lstsq(X1[tr], y[tr], rcond=None)[0]; p = X1 @ w
        for g in np.unique(ids[te]):
            for q in ('desc_opts', 'desc'):
                k = (ids == g) & (qs == q)
                if k.any():
                    res.append((q, y[k][np.argmax(p[k])], y[k][0], y[k].max()))
    w = np.linalg.lstsq(X1, y, rcond=None)[0]
    for q in ('desc_opts', 'desc'):
        r = np.array([x[1:] for x in res if x[0] == q])
        print(('post' if post else 'pre'), q, 'chosen %.3f  top1 %.3f  oracle %.3f' % tuple(r.mean(0)))
    return w.tolist()


model = dict(pre=fit(False), post=fit(True))
json.dump(model, open(f'{E.R}/data/splits/webshop_scorer.json', 'w'))
print('weights', model)
