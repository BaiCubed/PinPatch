"""P1-1 on ALFWorld: verified procedure evolution, same protocol as ws_evolve.py (pre-registered PROGRESS.md 2026-11-18).

  alf_evolve.py splits                 -> data/splits/alfworld_train_{evo,ho,lab}.json (train[0:1200], [1200:1800], [1800:2360])
  alf_evolve.py replay BASE_RUN URL    -> runs/p11a/replay.jsonl: each failed base episode rerun under the 7 other configs
  alf_evolve.py evolve BASE_RUN URL    -> runs/p11a/patches.json: per 'task_type|object', rank configs by repairs, replay
                                          gate on that key's successful base episodes (any break -> rejected), keep best
  alf_evolve.py random BASE_RUN        -> runs/p11a/patches_random.json: same keys, a random non-default config each
URL = the Qwen serial vLLM endpoint (meta_alf asks the LLM only when stuck)."""
import itertools, json, os, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import run_eval as E

R = E.R; OUT = f'{R}/runs/p11a'; os.makedirs(OUT, exist_ok=True)
CFGS = [dict(search=a, take=b, target=c) for a, b, c in itertools.product(('prior', 'plain'), ('first', 'last'), ('low', 'high'))]
ALT = [c for c in CFGS if c != E.ALF_DEFAULT]
key = lambda c: f"{c['search']},{c['take']},{c['target']}"


def args(url):
    return SimpleNamespace(env='alfworld', max_steps=30, llm_url=url, seed=0)


def run(i, cfg, url):
    r = E.meta_alf(i, args(url), cfg=cfg)
    return dict(id=i, cfg=key(cfg), success=r['success'], key=r['key'])


def base(run_id):
    return [json.loads(l) for l in open(f'{R}/runs/{run_id}/episodes.jsonl')]


def pmap(fn, jobs, w=8):
    with ThreadPoolExecutor(w) as ex:
        return list(ex.map(lambda j: fn(*j), jobs))


def splits():
    tr = json.load(open(f'{R}/data/splits/alfworld.json'))['train']
    for name, (a, b) in dict(evo=(0, 1200), ho=(1200, 1800), lab=(1800, 2360)).items():
        json.dump(tr[a:b], open(f'{R}/data/splits/alfworld_train_{name}.json', 'w')); print(name, len(tr[a:b]))


def replay(run_id, url):
    fails = [r['id'] for r in base(run_id) if not r['success']]
    f = f'{OUT}/replay.jsonl'; done = set()
    if os.path.exists(f):
        done = {(x['id'], x['cfg']) for x in map(json.loads, open(f))}
    jobs = [(i, c, url) for i in fails for c in ALT if (i, key(c)) not in done]
    print('failures', len(fails), 'replays to run', len(jobs), flush=True)
    with open(f, 'a') as fo:
        for k in range(0, len(jobs), 64):
            for x in pmap(run, jobs[k:k + 64]):
                fo.write(json.dumps(x) + '\n'); fo.flush()


def evolve(run_id, url):
    B = base(run_id); kk = {r['id']: r['key'] for r in B}
    rep = [json.loads(l) for l in open(f'{OUT}/replay.jsonl')]
    fixes = defaultdict(Counter)
    for x in rep:
        if x['success']:
            fixes[kk[x['id']]][x['cfg']] += 1
    succ = defaultdict(list)
    for r in B:
        if r['success']:
            succ[r['key']].append(r['id'])
    patches, log = {}, []
    for c, cnt in sorted(fixes.items(), key=lambda z: -sum(z[1].values())):
        for k, nfix in cnt.most_common():
            cfg = dict(zip(('search', 'take', 'target'), k.split(',')))
            res = pmap(run, [(i, cfg, url) for i in succ[c]])
            broken = sum(not r['success'] for r in res)
            log.append(dict(key=c, cfg=k, repairs=nfix, checked=len(res), broken=broken))
            if broken == 0:
                patches[c] = cfg; break
    json.dump(patches, open(f'{OUT}/patches.json', 'w'), indent=1)
    with open(f'{OUT}/evolve_log.jsonl', 'w') as fo:
        for x in log:
            fo.write(json.dumps(x) + '\n')
    print('failures', sum(not r['success'] for r in B), 'repairable', len({x['id'] for x in rep if x['success']}),
          'keys with a repair', len(fixes), 'patched', len(patches), 'repairs covered', sum(fixes[c][key(p)] for c, p in patches.items()))


def rand():
    P = json.load(open(f'{OUT}/patches.json')); rng = random.Random(20261118)
    out = {c: rng.choice(ALT) for c in sorted(P)}
    json.dump(out, open(f'{OUT}/patches_random.json', 'w'), indent=1); print('random patches', len(out))


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'splits':
        splits()
    elif c == 'replay':
        replay(sys.argv[2], sys.argv[3])
    elif c == 'evolve':
        evolve(sys.argv[2], sys.argv[3])
    else:
        rand()
