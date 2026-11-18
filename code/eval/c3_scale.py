"""C3 at scale (ScienceWorld TRAIN only): more failures -> more demo patches -> coverage of the demos that dev/test
tasks retrieve. Only the edits a method would try within its budget are run, so credit order decides the cost.

  c3_scale.py base N            base runs (deployed system) on ~N new stratified TRAIN tasks -> DIR/base.jsonl
  c3_scale.py orders            credit orders for every failure (GPU) -> DIR/orders.json
  c3_scale.py search B          run the union of the first B edits of gate / random / recency -> DIR/edits.jsonl
DIR = runs/repair2. Then: C3_DIR=runs/repair2 evolve.py METHOD B OUT."""
import json, os, random, sys
from concurrent.futures import ThreadPoolExecutor

import repair
import run_eval as E

DIR = os.environ.get('C3_DIR', f'{E.R}/runs/repair2')


def base(n):
    os.makedirs(DIR, exist_ok=True)
    g = json.load(open(f'{E.R}/data/splits/sciworld_games.json'))
    tr = json.load(open(f'{E.R}/data/splits/sciworld.json'))['train']
    used = set(json.load(open(f'{E.R}/data/splits/sciworld_train_heldout.json')))
    used |= {json.loads(l)['id'] for l in open(f'{E.R}/runs/repair/base.jsonl')}
    by = {}
    for i in tr:
        if i not in used:
            by.setdefault(g[str(i)], []).append(i)
    rng = random.Random(20261112); pick = []
    for t, v in sorted(by.items()):
        rng.shuffle(v); pick += v[: n // len(by)]
    out = f'{DIR}/base.jsonl'
    done = {json.loads(l)['id'] for l in open(out)} if os.path.exists(out) else set()
    with open(out, 'a') as f, ThreadPoolExecutor(32) as ex:
        for r in ex.map(repair.run, [i for i in pick if i not in done]):
            f.write(json.dumps(r) + '\n'); f.flush()
    print('BASE2_DONE', len(pick))


def base_ids(files):
    """Round k+1: rerun the base of earlier rounds (their task ids) with the round-k library (PATCHES env)."""
    os.makedirs(DIR, exist_ok=True)
    ids = sorted({json.loads(l)['id'] for f in files for l in open(f)})
    out = f'{DIR}/base.jsonl'
    done = {json.loads(l)['id'] for l in open(out)} if os.path.exists(out) else set()
    with open(out, 'a') as f, ThreadPoolExecutor(32) as ex:
        for r in ex.map(repair.run, [i for i in ids if i not in done]):
            f.write(json.dumps(r) + '\n'); f.flush()
    print('BASEIDS_DONE', len(ids))


def orders():
    import numpy as np, torch
    import lch, repair_eval as RE
    B = [json.loads(l) for l in open(f'{DIR}/base.jsonl')]
    fails = [r for r in B if not r['success']]
    env = E.ENVS['sciworld'](); tasks = {r['id']: env.reset(r['id']).split('\n')[0] for r in fails}; env.close()
    head = torch.load(f'{E.R}/runs/lch/v1.pt', weights_only=False); gate = RE.train_gate(); out = {}
    for r in fails:
        H = torch.tensor(lch.encode(RE.step_texts(r, tasks[r['id']]), (20,), bs=8)[20])
        pg = torch.sigmoid(((H - gate['mu']) / gate['sd']) @ gate['w'] + gate['b']).numpy()
        pl = torch.sigmoid(((H - head['mu']) / head['sd']) @ head['w'] + head['b']).numpy()
        prov = r['prov'] or []; acts = [a for a, _ in r['trace']]

        def comp(e, p):
            if 'skip_plan' in e:
                v = [p[k] for k, pk in enumerate(prov) if pk == e['skip_plan']]
            elif 'drop_sub' in e:
                y = (r['subs'] or {}).get(e['drop_sub'], e['drop_sub']); v = [p[k] for k, a in enumerate(acts) if y in a]
            else:
                return float(np.mean(p)) if len(p) else 0.0
            return float(max(v)) if v else 0.0
        c = repair.candidates(r)
        out[str(r['id'])] = dict(gate=sorted(c, key=lambda e: -comp(e, pg)), lch=sorted(c, key=lambda e: -comp(e, pl)),
                                 recency=sorted(c, key=lambda e: -e.get('skip_plan', -1)),
                                 random=random.Random(r['id']).sample(c, len(c)))
    json.dump(out, open(f'{DIR}/orders.json', 'w'))
    print('ORDERS_DONE', len(out))


def search(b):
    O = json.load(open(f'{DIR}/orders.json'))
    jobs, seen = [], set()
    out = f'{DIR}/edits.jsonl'
    if os.path.exists(out):
        for l in open(out):
            x = json.loads(l); seen.add((x['id'], json.dumps(x['edit'], sort_keys=True)))
    for i, o in O.items():
        for m in os.environ.get('C3_METHODS', 'gate,random,recency').split(','):
            for e in o[m][:b]:
                key = (int(i), json.dumps(e, sort_keys=True))
                if key not in seen:
                    seen.add(key); jobs.append(key)
    print(len(jobs), 'edit runs', flush=True)
    with open(out, 'a') as f, ThreadPoolExecutor(32) as ex:
        for k, r in enumerate(ex.map(lambda j: repair.run(j[0], json.loads(j[1])), jobs)):
            r.pop('trace', None); f.write(json.dumps(r) + '\n'); f.flush()
    print('SEARCH_DONE')


if __name__ == '__main__':
    {'base': lambda: base(int(sys.argv[2])), 'base_ids': lambda: base_ids(sys.argv[2:]), 'orders': orders,
     'search': lambda: search(int(sys.argv[2]))}[sys.argv[1]]()
