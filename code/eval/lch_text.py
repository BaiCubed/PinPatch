"""R4: constraint baselines that do NOT read hidden states. Same labelled data as the v1 head (runs/lch/v1.jsonl:
same TRAIN episodes, same outcome labels, same input text E.lch_text), same 20% held-out episode split (seed 20261106),
same deployed threshold rule as v1 (maximise held-out recall subject to false-block <= 0.2%), then refit on all.

  tfidf : word 1-2 gram TF-IDF of the head's input text (task + last 3 steps + candidate action), logistic regression
          (same loss / pos_weight / AdamW / 300 epochs as lch.fit; lr 1e-2 because the inputs are sparse)
  lookup: action-outcome table, p = (violations + 1) / (count + 2) for the lower-cased candidate action string;
          unseen actions get the TRAIN base rate

  lch_text.py train VARIANT OUT.pt         -> head + held-out report (auc, recall, false_block)
  lch_text.py serve OUT.pt PORT            -> HTTP /score {texts} -> {p, tau}  (drop-in for lch.py serve; CPU only)
Use with META_SHIELD=learned|both, LCH_URL=http://127.0.0.1:PORT and LCH_TAU unset (the threshold comes from the head)."""
import json, math, random, re, sys
from collections import Counter

import numpy as np

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')); DATA = f'{R}/runs/lch/v1.jsonl'
FB_MAX = 0.002


def act_of(text):
    return text.rsplit('Candidate action:', 1)[-1].strip().lower()


def grams(text):
    w = re.findall(r'[a-z0-9]+', text.lower())
    return w + [a + ' ' + b for a, b in zip(w, w[1:])]


class Tfidf:
    def __init__(self, texts, min_df=2):
        df = Counter(g for t in texts for g in set(grams(t)))
        self.voc = {g: i for i, g in enumerate(sorted(g for g, c in df.items() if c >= min_df))}
        n = len(texts); self.idf = np.zeros(len(self.voc), dtype=np.float32)
        for g, i in self.voc.items():
            self.idf[i] = math.log((1 + n) / (1 + df[g])) + 1

    def X(self, texts):
        import torch
        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            c = Counter(self.voc[g] for g in grams(t) if g in self.voc)
            v = np.array([n * self.idf[i] for i, n in c.items()], dtype=np.float32); nrm = float(np.linalg.norm(v)) or 1.0
            for (i, _), x in zip(c.items(), v):
                rows.append(r); cols.append(i); vals.append(x / nrm)
        return torch.sparse_coo_tensor([rows, cols], vals, (len(texts), len(self.voc))).coalesce()


def fit_tfidf(texts, y, epochs=300, wd=1e-3):
    import torch
    tf = Tfidf(texts); X = tf.X(texts); y = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(len(tf.voc), requires_grad=True); b = torch.zeros(1, requires_grad=True)
    pos = max(1.0, float((1 - y).sum() / max(1, y.sum())))
    opt = torch.optim.AdamW([w, b], lr=1e-2, weight_decay=wd); lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(torch.sparse.mm(X, w.unsqueeze(1)).squeeze(1) + b, y); loss.backward(); opt.step()
    return dict(kind='tfidf', tf=tf, w=w.detach(), b=b.detach())


def fit_lookup(texts, y):
    c, v = Counter(), Counter()
    for t, l in zip(texts, y):
        a = act_of(t); c[a] += 1; v[a] += int(l)
    return dict(kind='lookup', table={a: (v[a] + 1) / (c[a] + 2) for a in c}, prior=float(np.mean(y)))


def predict(h, texts):
    import torch
    if h['kind'] == 'lookup':
        return np.array([h['table'].get(act_of(t), h['prior']) for t in texts])
    return torch.sigmoid(torch.sparse.mm(h['tf'].X(texts), h['w'].unsqueeze(1)).squeeze(1) + h['b']).numpy()


def strict_tau(p, y):
    best = (0.0, 1.0)  # (recall, tau); tau 1.0 = never block
    neg = max(1, int((y == 0).sum())); pos_n = max(1, int((y == 1).sum()))
    for t in np.unique(np.round(p, 4)):
        fb = ((p > t) & (y == 0)).sum() / neg
        if fb <= FB_MAX:
            best = max(best, (((p > t) & (y == 1)).sum() / pos_n, float(t)))
    return best[1], best[0]


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def train(variant, out):
    import torch
    rows = [json.loads(l) for l in open(DATA)]
    texts = [r['text'] for r in rows]; y = np.array([r['y'] for r in rows]); ep = [r['ep'] for r in rows]
    eps = sorted(set(ep)); random.Random(20261106).shuffle(eps); hold = set(eps[: len(eps) // 5])
    te = np.array([e in hold for e in ep]); tr = ~te
    fit = fit_tfidf if variant == 'tfidf' else fit_lookup
    h = fit([t for t, m in zip(texts, tr) if m], y[tr]); p = predict(h, [t for t, m in zip(texts, te) if m])
    tau, rec = strict_tau(p, y[te]); fb = ((p > tau) & (y[te] == 0)).sum() / max(1, (y[te] == 0).sum())
    rep = dict(variant=variant, auc=round(auc(p, y[te]), 3), tau=tau, recall=round(float(rec), 3), false_block=round(float(fb), 4),
               n=int(len(y)), n_pos=int(y.sum()), n_heldout=int(te.sum()))
    h = fit(texts, y); h['tau'] = tau; h['report'] = rep
    torch.save(h, out); print(json.dumps(rep), '->', out)


def serve(head, port):
    import torch, uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    import __main__
    __main__.Tfidf = Tfidf  # unpickling the vectoriser
    h = torch.load(head, weights_only=False); app = FastAPI()

    class Q(BaseModel):
        texts: list

    @app.post('/score')
    def score(q: Q):
        return dict(p=[float(x) for x in predict(h, q.texts)], tau=h['tau'])

    uvicorn.run(app, host='127.0.0.1', port=int(port))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(sys.argv[2], sys.argv[3])
    else:
        serve(sys.argv[2], sys.argv[3])
