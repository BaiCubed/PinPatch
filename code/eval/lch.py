"""Plan B step 1: learned constraint head (LCH). Replaces the hand-written ScienceWorld focus shield with a head on
the frozen policy model's own hidden states (Qwen2.5-7B-Instruct), trained on TRAIN interaction only.

Labels (no hand-written rule involved): run the controller with no shield on TRAIN tasks (leave-one-out demo
retrieval); an action after which the episode ended unsuccessfully before the step cap is a violation (1); every
other executed action is allowed (0): it was carried out and the episode went on.

  lch.py data  RUN_DIR...  OUT.jsonl          build labelled examples
  lch.py feats IN.jsonl OUT.npz [layers]      last-token hidden states of the given layers
  lch.py train IN.npz OUT.pt                  logistic head per layer; pick layer and threshold on a held-out
                                              20% of episodes (threshold maximises recall - 3 * false-block rate)
  lch.py serve HEAD.pt [port]                 HTTP /score {texts} -> {p, tau}, used by META_SHIELD=learned
"""
import json, os, random, sys

import run_eval as E

LAYERS = (14, 20, 27)


def task_texts(ids):
    env = E.ENVS['sciworld'](); out = {}
    for i in ids:
        out[i] = env.reset(i).split('\n')[0]
    env.close()
    return out


def data(run_dirs, out):
    eps = []
    for d in run_dirs:
        for l in open(f'{d}/episodes.jsonl'):
            try:
                eps.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    tasks = task_texts(sorted({r['id'] for r in eps}))
    rows = []
    for r in eps:
        tr = r['trace']
        ended_bad = (not r['success']) and r['steps'] < 30 and len(tr) > 0
        for k, (act, _) in enumerate(tr):
            y = int(ended_bad and k == len(tr) - 1)  # 1: this action ended the episode in failure
            rows.append(dict(ep=r['id'], k=k, y=y, act=act, text=E.lch_text(tasks[r['id']], tr[:k], act)))
    with open(out, 'w') as f:
        for x in rows:
            f.write(json.dumps(x) + '\n')
    print(f'{len(eps)} episodes ({sum(r["success"] for r in eps)} success), {len(rows)} examples, '
          f'{sum(x["y"] for x in rows)} violations -> {out}')


_M = {}


def model():
    if 'm' not in _M:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(E.MODEL)
        m = AutoModelForCausalLM.from_pretrained(E.MODEL, torch_dtype=torch.bfloat16, device_map='cuda').eval()
        _M.update(m=m, tok=tok)
    return _M['m'], _M['tok']


def encode(texts, layers=LAYERS, bs=16):
    import torch
    m, tok = model(); tok.padding_side = 'left'
    out = {L: [] for L in layers}
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], return_tensors='pt', padding=True, truncation=True, max_length=768).to('cuda')
            hs = m(**b, output_hidden_states=True).hidden_states
            for L in layers:
                out[L].append(hs[L][:, -1, :].float().cpu())
    return {L: torch.cat(v).numpy() for L, v in out.items()}


def feats(inp, out, layers=LAYERS):
    import numpy as np
    rows = [json.loads(l) for l in open(inp)]
    H = encode([x['text'] for x in rows], layers)
    np.savez(out, y=np.array([x['y'] for x in rows]), ep=np.array([x['ep'] for x in rows]),
             **{f'h{L}': H[L] for L in layers})
    print('features', {L: H[L].shape for L in layers}, '->', out)


def fit(X, y, epochs=300, wd=1e-3):
    import torch
    X = torch.tensor(X); y = torch.tensor(y, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lin = torch.nn.Linear(X.shape[1], 1)
    pos = max(1.0, float((1 - y).sum() / max(1, y.sum())))
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(lin((X - mu) / sd).squeeze(1), y); loss.backward(); opt.step()
    return dict(w=lin.weight.detach().squeeze(0), b=lin.bias.detach(), mu=mu, sd=sd)


def prob(h, X):
    import torch
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.sigmoid(((X - h['mu']) / h['sd']) @ h['w'] + h['b']).numpy()


def auc(p, y):
    import numpy as np
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def pick_tau(p, y):
    import numpy as np
    best = (-9, 0.5)
    for t in np.unique(np.round(p, 3)):
        rec = ((p > t) & (y == 1)).sum() / max(1, (y == 1).sum()); fpr = ((p > t) & (y == 0)).sum() / max(1, (y == 0).sum())
        best = max(best, (rec - 3 * fpr, float(t)))
    return best[1], best[0]


def train(inp, out):
    import numpy as np, torch
    d = np.load(inp); y = d['y']; ep = d['ep']
    eps = sorted(set(ep.tolist())); random.Random(20261106).shuffle(eps); hold = set(eps[: len(eps) // 5])
    te = np.array([e in hold for e in ep]); trm = ~te
    rep, best = {}, None
    for key in [k for k in d.files if k.startswith('h')]:
        h = fit(d[key][trm], y[trm]); p = prob(h, d[key][te])
        tau, j = pick_tau(p, y[te]); a = auc(p, y[te])
        rec = ((p > tau) & (y[te] == 1)).sum() / max(1, (y[te] == 1).sum())
        fpr = ((p > tau) & (y[te] == 0)).sum() / max(1, (y[te] == 0).sum())
        rep[key] = dict(auc=round(a, 3), tau=tau, recall=round(float(rec), 3), false_block=round(float(fpr), 4))
        if best is None or a > best[1]:
            best = (key, a, tau)
    key, a, tau = best
    h = fit(d[key], y)  # refit on all TRAIN examples with the chosen layer; threshold from the held-out split
    torch.save(dict(layer=int(key[1:]), tau=tau, **h, report=rep, n=int(len(y)), n_pos=int(y.sum())), out)
    print(json.dumps(rep), '\nchosen', key, 'tau', tau, '->', out)


def serve(head, port=36110):
    import torch, uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    h = torch.load(head, weights_only=False); L = h['layer']; app = FastAPI()

    class Q(BaseModel):
        texts: list

    @app.post('/score')
    def score(q: Q):
        H = encode(q.texts, (L,), bs=1)[L]
        return dict(p=[float(x) for x in prob(h, H)], tau=h['tau'])

    model(); uvicorn.run(app, host='127.0.0.1', port=int(port))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'data':
        data(sys.argv[2:-1], sys.argv[-1])
    elif cmd == 'feats':
        feats(sys.argv[2], sys.argv[3], tuple(int(x) for x in sys.argv[4].split(',')) if len(sys.argv) > 4 else LAYERS)
    elif cmd == 'train':
        train(sys.argv[2], sys.argv[3])
    elif cmd == 'serve':
        serve(sys.argv[2], *(sys.argv[3:4]))
