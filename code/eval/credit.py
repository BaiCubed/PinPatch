"""E1: credit assignment over the steps of executable-procedure trajectories (ScienceWorld, TRAIN only).

Ground truth (counterfactual, env replay, no LLM): in a failed episode, step t is critical if replaying the same
action sequence with step t removed ends better (higher final score, or no longer ends early in failure).
Credit methods are scored on how well they rank those steps from the trajectory alone.

  credit.py cf  OUT.jsonl N [SCIWORLD_URL...]   counterfactual labels for N sampled failed episodes
  credit.py eval CF.jsonl                       localisation of: last step, LCH violation score, sparse outcome
                                                gate (episode outcome only, no step labels), random
Inputs: the no-shield leave-one-out TRAIN runs (runs/lch1_sci_noshield_train_{a,b,c,d}) and their per-step
features (runs/lch/v1.jsonl + v1.npz, one row per executed action, ordered by episode then step)."""
import json, os, random, sys
from concurrent.futures import ThreadPoolExecutor

import run_eval as E

RUNS = [f'{E.R}/runs/lch1_sci_noshield_train_{h}' for h in 'abcd']
SKIP = ('look around', 'inventory', 'wait', 'wait1', 'look at ')


def episodes():
    out = []
    for d in RUNS:
        for l in open(f'{d}/episodes.jsonl'):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def replay(idx, acts, url):
    os.environ['SCIWORLD_URL'] = url
    env = E.SciWorld.__new__(E.SciWorld); env.url = url
    env.id = E.requests.post(url + '/create', timeout=300).json()['id']
    env.reset(idx); reward, done, n = 0, False, 0
    for a in acts:
        if n >= 30 or done:
            break
        o, reward, done = env.step(a); n += 1
        if o.startswith('Ambiguous request'):
            o, reward, done = env.step('0')
    env.close()
    return reward, done and reward < 100 and n < 30


def cf(out, n, urls):
    eps = [e for e in episodes() if not e['success']]
    random.Random(20261110).shuffle(eps)
    early = [e for e in eps if e['steps'] < 30][: n // 2]; cap = [e for e in eps if e['steps'] >= 30][: n - n // 2]
    jobs = []
    for e in early + cap:
        acts = [a for a, o in e['trace']]
        cand = [t for t, a in enumerate(acts) if not a.startswith(SKIP)]
        if len(cand) > 15:  # evenly spaced, always keeping the last step
            cand = sorted(set(cand[:: max(1, len(cand) // 14)][:14] + [cand[-1]]))
        jobs.append((e, acts, cand))
    k = [0]

    def run(job):
        e, acts, cand = job
        url = urls[k[0] % len(urls)]; k[0] += 1
        base, base_early = replay(e['id'], acts, url)
        rows = []
        for t in cand:
            s, ea = replay(e['id'], acts[:t] + acts[t + 1:], url)
            rows.append(dict(ep=e['id'], t=t, act=acts[t], n=len(acts), orig=base, orig_early=base_early,
                             cf=s, cf_early=ea, delta=s - base,
                             critical=int(s > base or (base_early and not ea))))
        return rows

    with open(out, 'w') as f, ThreadPoolExecutor(12) as ex:
        for i, rows in enumerate(ex.map(run, jobs)):
            for r in rows:
                f.write(json.dumps(r) + '\n')
            if i % 20 == 0:
                print(i, len(jobs), flush=True)
    print('CF_DONE', out)


def gate_scores():
    """Sparse outcome gate: per-step failure gates g_t = sigmoid(w.h_t + b) on the policy's hidden states (layer 20),
    episode P(fail) = 1 - prod(1 - g_t) (noisy-or), trained on episode outcomes only, with an L1 penalty on g."""
    import numpy as np, torch
    rows = [json.loads(l) for l in open(f'{E.R}/runs/lch/v1.jsonl')]
    X = torch.tensor(np.load(f'{E.R}/runs/lch/v1.npz')['h20']); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    succ = {e['id']: e['success'] for e in episodes()}
    eps = sorted({r['ep'] for r in rows}); idx = {e: i for i, e in enumerate(eps)}
    E_ = torch.tensor([idx[r['ep']] for r in rows]); Y = torch.tensor([0.0 if succ[e] else 1.0 for e in eps])
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.full((1,), -4.0, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-3)
    for it in range(600):
        g = torch.sigmoid(X @ w + b)
        logsurv = torch.zeros(len(eps)).index_add(0, E_, torch.log1p(-g.clamp(max=1 - 1e-6)))
        p = (1 - torch.exp(logsurv)).clamp(1e-6, 1 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(p, Y) + 0.01 * g.mean() + 1e-4 * w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    g = torch.sigmoid(X @ w + b).detach().numpy()
    return {(r['ep'], r['k']): float(x) for r, x in zip(rows, g)}


def lch_scores():
    import numpy as np, torch
    rows = [json.loads(l) for l in open(f'{E.R}/runs/lch/v1.jsonl')]
    h = torch.load(f'{E.R}/runs/lch/v1.pt', weights_only=False)
    X = torch.tensor(np.load(f'{E.R}/runs/lch/v1.npz')[f"h{h['layer']}"])
    p = torch.sigmoid(((X - h['mu']) / h['sd']) @ h['w'] + h['b']).numpy()
    return {(r['ep'], r['k']): float(x) for r, x in zip(rows, p)}


def evaluate(cf_path):
    import numpy as np
    by = {}
    for l in open(cf_path):
        r = json.loads(l); by.setdefault(r['ep'], []).append(r)
    by = {e: v for e, v in by.items() if any(x['critical'] for x in v)}
    S = dict(gate=gate_scores(), lch=lch_scores())
    rng = random.Random(0)
    res = {}
    for name in ('last', 'lch', 'gate', 'random'):
        for kind in ('early', 'cap', 'all'):
            h1 = h3 = n = 0; mrr = 0.0
            for e, v in by.items():
                is_early = v[0]['orig_early']
                if kind == 'early' and not is_early or kind == 'cap' and is_early:
                    continue
                if name == 'last':
                    sc = [x['t'] for x in v]
                elif name == 'random':
                    sc = [rng.random() for x in v]
                else:
                    sc = [S[name].get((e, x['t']), 0.0) for x in v]
                order = [v[i] for i in np.argsort(sc)[::-1]]
                rank = next(i for i, x in enumerate(order) if x['critical'])
                h1 += rank == 0; h3 += rank < 3; mrr += 1 / (rank + 1); n += 1
            res[(name, kind)] = (n, h1 / max(1, n), h3 / max(1, n), mrr / max(1, n))
    for kind in ('early', 'cap', 'all'):
        print(f'-- {kind} failures (episodes with >=1 critical step: {res[("last", kind)][0]})')
        for name in ('last', 'lch', 'gate', 'random'):
            n, a, b, c = res[(name, kind)]
            print(f'   {name:7s} hit@1 {a:.3f}  hit@3 {b:.3f}  MRR {c:.3f}')
    crit = [x for v in by.values() for x in v if x['critical']]
    print('critical steps per episode %.2f; share that are the last step %.3f' % (
        len(crit) / max(1, len(by)), sum(x['t'] == x['n'] - 1 for x in crit) / max(1, len(crit))))


if __name__ == '__main__':
    if sys.argv[1] == 'cf':
        cf(sys.argv[2], int(sys.argv[3]), sys.argv[4:] or ['http://127.0.0.1:36102', 'http://127.0.0.1:36112'])
    else:
        evaluate(sys.argv[2])
