"""P1-1: verified procedure evolution on WebShop (pre-registered PROGRESS.md 2026-11-18). TRAIN only, no LLM.

  ws_evolve.py splits                -> data/splits/webshop_train_{evo,ho,lab}.json (fixed id ranges of the train list)
  ws_evolve.py replay BASE_RUN       -> runs/p11/replay.jsonl: every failed base episode of BASE_RUN rerun under each of
                                        the 11 non-default configurations (deterministic, resettable)
  ws_evolve.py evolve BASE_RUN       -> runs/p11/patches.json (+ candidates, validation log): per category, rank configs
                                        by repairs - breaks among replays, validate on the category's successful base
                                        episodes (any break -> rejected), keep the best surviving config
  ws_evolve.py random BASE_RUN       -> runs/p11/patches_random.json: same categories, a random non-default config each"""
import itertools, json, os, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import run_eval as E

R = E.R; OUT = f'{R}/runs/p11'; os.makedirs(OUT, exist_ok=True)
CFGS = [dict(query=q, order=o, commit=c) for q, o, c in itertools.product(('desc_opts', 'desc', 'full'), ('price', 'search'), ('first', 'best'))]
ALT = [c for c in CFGS if c != E.WS_DEFAULT]
A = SimpleNamespace(env='webshop', max_steps=10)
key = lambda c: f"{c['query']},{c['order']},{c['commit']}"


def run(i, cfg):
    r = E.meta_ws(i, A, cfg=cfg)
    return dict(id=i, cfg=key(cfg), success=r['success'], score=r['score'], category=r['category'])


def base(run_id):
    return [json.loads(l) for l in open(f'{R}/runs/{run_id}/episodes.jsonl')]


def splits():
    tr = json.load(open(f'{R}/data/splits/webshop.json'))['train']
    for name, (a, b) in dict(evo=(300, 1800), ho=(1800, 2400), lab=(2400, 4400)).items():
        json.dump(tr[a:b], open(f'{R}/data/splits/webshop_train_{name}.json', 'w')); print(name, b - a)


def pmap(fn, jobs, w=16):
    with ThreadPoolExecutor(w) as ex:
        return list(ex.map(lambda j: fn(*j), jobs))


def replay(run_id):
    fails = [r['id'] for r in base(run_id) if not r['success']]
    f = f'{OUT}/replay.jsonl'; done = set()
    if os.path.exists(f):
        done = {(x['id'], x['cfg']) for x in map(json.loads, open(f))}
    jobs = [(i, c) for i in fails for c in ALT if (i, key(c)) not in done]
    print('failures', len(fails), 'replays to run', len(jobs))
    with open(f, 'a') as fo:
        for k in range(0, len(jobs), 256):
            for x in pmap(run, jobs[k:k + 256]):
                fo.write(json.dumps(x) + '\n'); fo.flush()


def evolve(run_id):
    B = base(run_id); cat = {r['id']: r['category'] for r in B}
    rep = [json.loads(l) for l in open(f'{OUT}/replay.jsonl')]
    fixes = defaultdict(Counter)
    for x in rep:
        if x['success']:
            fixes[cat[x['id']]][x['cfg']] += 1
    succ = defaultdict(list)
    for r in B:
        if r['success']:
            succ[r['category']].append(r['id'])
    patches, log = {}, []
    for c, cnt in sorted(fixes.items(), key=lambda z: -sum(z[1].values())):
        for k, nfix in cnt.most_common():
            cfg = dict(zip(('query', 'order', 'commit'), k.split(',')))
            res = pmap(run, [(i, cfg) for i in succ[c]])  # replay gate on the category's successes
            broken = sum(not r['success'] for r in res)
            log.append(dict(category=c, cfg=k, repairs=nfix, checked=len(res), broken=broken))
            if broken == 0:
                patches[c] = cfg; break
    json.dump(patches, open(f'{OUT}/patches.json', 'w'), indent=1)
    with open(f'{OUT}/evolve_log.jsonl', 'w') as fo:
        for x in log:
            fo.write(json.dumps(x) + '\n')
    print('categories with a repair', len(fixes), 'patched', len(patches),
          'repairs covered', sum(fixes[c][key(p)] for c, p in patches.items()))


def rand(run_id):
    P = json.load(open(f'{OUT}/patches.json')); rng = random.Random(20261118)
    out = {c: rng.choice(ALT) for c in sorted(P)}
    json.dump(out, open(f'{OUT}/patches_random.json', 'w'), indent=1); print('random patches', len(out))


if __name__ == '__main__':
    cmd = sys.argv[1]
    dict(splits=splits, replay=lambda: replay(sys.argv[2]), evolve=lambda: evolve(sys.argv[2]),
         random=lambda: rand(sys.argv[2]))[cmd]()
