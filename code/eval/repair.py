"""C2/E3 on ScienceWorld (TRAIN only): credit-guided repair of executable procedures.

  repair.py base N OUT.jsonl        run the deployed system (learned+hand guard) on N train tasks, provenance on
  repair.py edits BASE.jsonl OUT    for each failed episode, run every candidate procedure edit once:
                                    demo_rank 1|2, drop_sub <each lifted slot>, skip_plan <each executed plan step>
A repair is an edit that turns the failure into a success. Credit orderings are compared on how many tries they
need before the first repair (repair_eval.py). Serial servers (deterministic), leave-one-out demo retrieval."""
import argparse, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('LOO_DEMO', '1'); os.environ.setdefault('META_SHIELD', 'both'); os.environ.setdefault('LCH_TAU', '0.9761')
os.environ['LOG_PROV'] = '1'
import run_eval as E

PORTS = [int(x) for x in os.environ.get('REPAIR_PORTS', '8123,8133,8124,8134,8125,8135,8126,8136').split(',')]
SCI = ['http://127.0.0.1:36102', 'http://127.0.0.1:36112']
_lock = threading.Lock(); _k = [0]; _tl = threading.local()


def _sci_init(self):  # per-thread env server: the URL lives on the instance, so threads never share it
    self.url = getattr(_tl, 'sci', SCI[0])
    self.id = E.requests.post(self.url + '/create', timeout=300).json()['id']


E.SciWorld.__init__ = _sci_init


def args_for(edit=None):
    with _lock:
        k = _k[0]; _k[0] += 1
    return argparse.Namespace(env='sciworld', llm_url=f'http://127.0.0.1:{PORTS[k % len(PORTS)]}/v1', seed=0,
                              max_steps=30, edit=edit), SCI[k % len(SCI)]


def run(idx, edit=None):
    a, sci = args_for(edit)
    _tl.sci = sci  # both env servers are identical AgentGym ScienceWorld servers
    r = E.meta_replay(idx, a)
    return dict(id=idx, edit=edit or {}, success=r['success'], score=r['score'], steps=r['steps'],
                trace=r['trace'], prov=r.get('prov'), plan=r.get('plan'), subs=r.get('subs_used'), demo=r['demo'])


def base(n, out):
    g = json.load(open(f'{E.R}/data/splits/sciworld_games.json'))
    ids = json.load(open(f'{E.R}/data/splits/sciworld_train_lch.json'))
    by = {}
    for i in ids:
        by.setdefault(g[str(i)], []).append(i)
    rng = random.Random(20261111); pick = []
    for t, v in sorted(by.items()):
        rng.shuffle(v); pick += v[: max(1, n // len(by))]
    done = {json.loads(l)['id'] for l in open(out)} if os.path.exists(out) else set()
    with open(out, 'a') as f, ThreadPoolExecutor(32) as ex:
        for r in ex.map(run, [i for i in pick if i not in done]):
            f.write(json.dumps(r) + '\n'); f.flush()
    print('BASE_DONE', out)


def candidates(r):
    c = [dict(demo_rank=1), dict(demo_rank=2)] + [dict(drop_sub=x) for x in (r['subs'] or {})]
    used = sorted({k for k in (r['prov'] or []) if k >= 0})
    c += [dict(skip_plan=k) for k in used]
    return c


def edits(base_path, out):
    B = [json.loads(l) for l in open(base_path)]
    fails = [r for r in B if not r['success']]
    done = set()
    if os.path.exists(out):
        for l in open(out):
            x = json.loads(l); done.add((x['id'], json.dumps(x['edit'], sort_keys=True)))
    jobs = [(r['id'], e) for r in fails for e in candidates(r) if (r['id'], json.dumps(e, sort_keys=True)) not in done]
    print(f'{len(fails)} failed of {len(B)}; {len(jobs)} edit runs', flush=True)
    with open(out, 'a') as f, ThreadPoolExecutor(32) as ex:
        for k, r in enumerate(ex.map(lambda j: run(*j), jobs)):
            r.pop('trace', None); f.write(json.dumps(r) + '\n'); f.flush()
            if k % 100 == 0:
                print(k, flush=True)
    print('EDITS_DONE', out)


if __name__ == '__main__':
    if sys.argv[1] == 'base':
        base(int(sys.argv[2]), sys.argv[3])
    else:
        edits(sys.argv[2], sys.argv[3])
