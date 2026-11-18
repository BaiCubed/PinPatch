"""C4 (ScienceWorld TRAIN only): evolution in a LARGER procedure-edit space, where credit should matter.

Atoms on a failed episode's procedure (indices = the patched plan the episode ran, as provenance reports them):
  skip k | swap k (k <-> k+1) | dup k (repeat step k) for every executed plan step k, drop_sub x for every lifted slot;
  plus demo_rank 1..3 (use a lower-ranked demo; not combined with other atoms).
Candidates = every single atom + every pair of atoms that touch disjoint plan indices (~10x the C3 space).

Pre-registered orderings (fixed before any outcome was seen):
  gate    best-first by credit: an edit's credit = MIN of its atoms' credits (an atom's credit = max sparse-gate
          probability over the steps it touches; demo_rank = mean over the episode); ties: fewer atoms, then
          skip < drop_sub < swap < dup, then text
  lch     the same with the learned constraint head instead of the gate
  random  uniform shuffle of all candidates (seed = task id)
  simple  singles in random order, then pairs in random order (a complexity prior with no credit)
Budget B tries per failure; the first repairing edit becomes a patch on the demo the episode used; patches are
validated by rerunning every base success that used that demo (as in evolve.py).

  evolve_x.py orders                 -> DIR/orders.json (GPU: encodes every failed episode)
  evolve_x.py search B               runs the union of the first B edits of C4_METHODS -> DIR/edits.jsonl
  evolve_x.py evolve METHOD B OUT    budgeted evolution + validation -> OUT (a PATCHES library)
DIR = C4_DIR (runs/repair4); base episodes = C4_BASE (comma list; default the two C3 round-1 bases)."""
import itertools, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor

import repair
import run_eval as E

R = E.R
DIR = os.environ.get('C4_DIR', f'{R}/runs/repair4')
BASES = os.environ.get('C4_BASE', f'{R}/runs/repair/base.jsonl,{R}/runs/repair2/base.jsonl').split(',')
LIB = json.load(open(os.environ['PATCHES_BASE'])) if os.environ.get('PATCHES_BASE') else {}
KIND = dict(skip=0, sub=1, swap=2, dup=3, demo=4)


def load_base():
    B = {}
    for f in BASES:
        for l in open(f):
            r = json.loads(l); B[r['id']] = r
    return B


def idx_of(a):
    return {a[1]} if a[0] in ('skip', 'dup') else {a[1], a[1] + 1} if a[0] == 'swap' else set()


def atoms(r):
    used = sorted({k for k in (r['prov'] or []) if k >= 0}); n = len(r['plan'] or [])
    out = [('sub', x) for x in sorted(r['subs'] or {})]
    for k in used:
        out += [('skip', k), ('dup', k)] + ([('swap', k)] if k + 1 < n else [])
    return out


def as_edit(at):
    if at[0][0] == 'demo':
        return dict(demo_rank=at[0][1])
    e = {}
    ops = sorted([list(a) for a in at if a[0] != 'sub'], key=lambda o: (o[1], o[0]))
    subs = sorted(a[1] for a in at if a[0] == 'sub')
    if ops:
        e['ops'] = ops
    if subs:
        e['drop_sub'] = subs
    return e


def candidates(r):
    A = atoms(r)
    c = [((('demo', k),)) for k in (1, 2, 3)] + [(a,) for a in A]
    c += [(a, b) for a, b in itertools.combinations(A, 2) if not (idx_of(a) & idx_of(b))]
    return c


def key(e):
    return json.dumps(e, sort_keys=True)


def orders():
    import numpy as np, torch
    import lch, repair_eval as RE
    os.makedirs(DIR, exist_ok=True)
    B = load_base(); fails = [r for r in B.values() if not r['success']]
    env = E.ENVS['sciworld'](); tasks = {r['id']: env.reset(r['id']).split('\n')[0] for r in fails}; env.close()
    head = torch.load(f'{R}/runs/lch/v1.pt', weights_only=False); gate = RE.train_gate(); out = {}; sizes = []
    for r in fails:
        H = torch.tensor(lch.encode(RE.step_texts(r, tasks[r['id']]), (20,), bs=8)[20])
        P = dict(gate=torch.sigmoid(((H - gate['mu']) / gate['sd']) @ gate['w'] + gate['b']).numpy(),
                 lch=torch.sigmoid(((H - head['mu']) / head['sd']) @ head['w'] + head['b']).numpy())
        prov = r['prov'] or []; acts = [a for a, _ in r['trace']]; subs = r['subs'] or {}

        def credit(a, p):
            if a[0] == 'demo':
                return float(np.mean(p)) if len(p) else 0.0
            if a[0] == 'sub':
                y = subs.get(a[1], a[1]); v = [p[k] for k, x in enumerate(acts) if y in x]
            else:
                v = [p[k] for k, pk in enumerate(prov) if pk in idx_of(a)]
            return float(max(v)) if v else 0.0
        C = candidates(r); sizes.append(len(C)); o = {}
        for m in ('gate', 'lch'):
            o[m] = [as_edit(c) for c in sorted(C, key=lambda c: (-min(credit(a, P[m]) for a in c), len(c),
                                                                 [KIND[a[0]] for a in c], str(c)))]
        rng = random.Random(r['id']); o['random'] = [as_edit(c) for c in rng.sample(C, len(C))]
        rng = random.Random(r['id'] + 1); s1 = [c for c in C if len(c) == 1]; s2 = [c for c in C if len(c) == 2]
        o['simple'] = [as_edit(c) for c in rng.sample(s1, len(s1)) + rng.sample(s2, len(s2))]
        out[str(r['id'])] = o
    json.dump(out, open(f'{DIR}/orders.json', 'w'))
    print('ORDERS_DONE', len(out), 'failures; candidates per failure: mean %.1f median %d min %d max %d'
          % (np.mean(sizes), int(np.median(sizes)), min(sizes), max(sizes)))


def search(b):
    O = json.load(open(f'{DIR}/orders.json')); out = f'{DIR}/edits.jsonl'; jobs, seen = [], set()
    if os.path.exists(out):
        for l in open(out):
            x = json.loads(l); seen.add((x['id'], key(x['edit'])))
    for i, o in O.items():
        for m in os.environ.get('C4_METHODS', 'gate,random,simple').split(','):
            for e in o[m][:b]:
                kk = (int(i), key(e))
                if kk not in seen:
                    seen.add(kk); jobs.append(kk)
    print(len(jobs), 'edit runs', flush=True)
    with open(out, 'a') as f, ThreadPoolExecutor(int(os.environ.get('C4_THREADS', 32))) as ex:
        for k, r in enumerate(ex.map(lambda j: repair.run(j[0], json.loads(j[1])), jobs)):
            r.pop('trace', None); f.write(json.dumps(r) + '\n'); f.flush()
            if k % 200 == 0:
                print(k, flush=True)
    print('SEARCH_DONE')


def merge_group(g, ops):
    """Add an edit's ops to a demo's op group for this round unless they touch indices already edited."""
    taken = set().union(*[idx_of(tuple(o)) for o in g]) if g else set()
    if any(idx_of(tuple(o)) & taken for o in ops):
        return False
    g += [list(o) for o in ops]; return True


def lib_with(lib, new):
    out = json.loads(json.dumps(lib))
    for d, p in new.items():
        q = out.setdefault(d, {})
        if p.get('exclude'):
            q['exclude'] = True
        if p.get('drop_sub'):
            q['drop_sub'] = sorted(set(q.get('drop_sub', [])) | set(p['drop_sub']))
        if p.get('ops'):
            q['ops'] = q.get('ops', []) + [p['ops']]
    return out


def evolve(method, B, outp):
    base = load_base(); ok = {}
    for l in open(f'{DIR}/edits.jsonl'):
        x = json.loads(l); ok[(x['id'], key(x['edit']))] = x['success']
    O = json.load(open(f'{DIR}/orders.json')); env = E.ENVS['sciworld']()
    patches, tries, found, missing = {}, 0, 0, 0
    for i, o in sorted(O.items(), key=lambda t: int(t[0])):
        i = int(i); seq = o[method][:B]; tries += len(seq)
        missing += sum((i, key(e)) not in ok for e in seq)
        hit = [e for e in seq if ok.get((i, key(e)))]
        if not hit:
            continue
        found += 1; e = hit[0]; dm = str(base[i]['demo']); p = patches.setdefault(dm, {})
        if 'demo_rank' in e:
            task = env.reset(i).split('\n')[0]
            for d in E.retrieve(task, 1 + e['demo_rank'], exclude=i)[: e['demo_rank']]:
                patches.setdefault(str(d['id']), {})['exclude'] = True
            continue
        if e.get('ops') and not merge_group(p.setdefault('ops', []), e['ops']):
            continue  # conflicts with an op already taken on this demo in this round
        if e.get('drop_sub'):
            p['drop_sub'] = sorted(set(p.get('drop_sub', [])) | set(e['drop_sub']))
    env.close()
    print(f'{method} B={B}: {tries} tries ({missing} not run), {found} failures repaired, {len(patches)} demos patched')
    keep, tmp = {}, outp + '.one'
    for dm, p in patches.items():
        succ = [i for i, r in base.items() if r['success'] and str(r['demo']) == dm]
        if not succ:
            keep[dm] = p; continue
        json.dump(lib_with(LIB, {dm: p}), open(tmp, 'w')); os.environ['PATCHES'] = tmp; E._PATCH.clear()
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda i: repair.run(i), succ))
        if all(r['success'] for r in res):
            keep[dm] = p
        else:
            print(f'  rejected patch on demo {dm}: breaks {sum(not r["success"] for r in res)}/{len(succ)} successes')
    json.dump(lib_with(LIB, keep), open(outp, 'w'), indent=1)
    print(f'kept {len(keep)} demo patches -> {outp}')


if __name__ == '__main__':
    {'orders': orders, 'search': lambda: search(int(sys.argv[2])),
     'evolve': lambda: evolve(sys.argv[2], int(sys.argv[3]), sys.argv[4])}[sys.argv[1]]()
