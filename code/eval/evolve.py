"""C3 on ScienceWorld (TRAIN only): credit-guided evolution of the procedure library under a search budget.

For each failed base episode, try candidate edits in a method's credit order, at most B tries (the edit outcomes
are already known from repair.py, so the search is simulated exactly). The first repairing edit becomes a patch on
the demo the episode used: demo_rank r -> exclude the r demos ranked above the one that worked; skip_plan k ->
drop step k of that demo's plan; drop_sub x -> do not substitute slot x for that demo. Patches are merged per demo
and validated: every base episode that succeeded with that demo is rerun with the patch, and a patch that breaks
any of them is dropped (replay gate, as in MetaAct, but checked in the environment).

  evolve.py METHOD B OUT.json      METHOD in gate | random | recency | lch | all (every repair, no budget)"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

import repair
import run_eval as E

R = E.R
D = os.environ.get('C3_DIR', f'{R}/runs/repair')  # runs/repair (first round) or runs/repair2 (scaled)
# round k+1: the library the base ran with; new skip indices refer to its already-shortened plans
LIB = json.load(open(os.environ['PATCHES_BASE'])) if os.environ.get('PATCHES_BASE') else {}


def orig_index(k, skipped):
    for s_ in sorted(skipped):
        if s_ <= k:
            k += 1
    return k


def merged(lib, new):
    out = json.loads(json.dumps(lib))
    for d, p in new.items():
        q = out.setdefault(d, {})
        if p.get('exclude'):
            q['exclude'] = True
        for k in ('skip_plan', 'drop_sub'):
            if p.get(k):
                q[k] = sorted(set(q.get(k, [])) | set(p[k]))
    return out


def proposed(method, B):
    base = {json.loads(l)['id']: json.loads(l) for l in open(f'{D}/base.jsonl')}
    ok = {}
    for l in open(f'{D}/edits.jsonl'):
        x = json.loads(l); ok[(x['id'], json.dumps(x['edit'], sort_keys=True))] = x['success']
    orders = json.load(open(f'{D}/orders.json'))
    env = E.ENVS['sciworld'](); patches, tries, found = {}, 0, 0
    for i, o in orders.items():
        i = int(i); seq = o['gate'] if method == 'all' else o[method]
        seq = seq if method == 'all' else seq[:B]
        hit = [e for e in seq if ok.get((i, json.dumps(e, sort_keys=True)))]
        tries += len(seq)
        if not hit:
            continue
        found += 1; e = hit[0]; dm = str(base[i]['demo']); p = patches.setdefault(dm, {})
        if 'demo_rank' in e:
            task = env.reset(i).split('\n')[0]
            ranked = E.retrieve(task, 1 + e['demo_rank'], exclude=i)
            for d in ranked[: e['demo_rank']]:
                patches.setdefault(str(d['id']), {})['exclude'] = True
        elif 'skip_plan' in e:
            k = orig_index(e['skip_plan'], LIB.get(dm, {}).get('skip_plan', []))
            p.setdefault('skip_plan', []); p['skip_plan'] = sorted(set(p['skip_plan'] + [k]))
        else:
            p.setdefault('drop_sub', []); p['drop_sub'] = sorted(set(p['drop_sub'] + [e['drop_sub']]))
    env.close()
    return patches, tries, found, base


def validate(patches, base, out):
    keep = {}
    tmp = out + '.one'
    for dm, p in patches.items():
        succ = [i for i, r in base.items() if r['success'] and str(r['demo']) == dm]
        if not succ:
            keep[dm] = p; continue
        json.dump(merged(LIB, {dm: p}), open(tmp, 'w')); os.environ['PATCHES'] = tmp; E._PATCH.clear()
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda i: repair.run(i), succ))
        if all(r['success'] for r in res):
            keep[dm] = p
        else:
            print(f'  rejected patch on demo {dm}: breaks {sum(not r["success"] for r in res)}/{len(succ)} successes')
    if LIB:
        os.environ['PATCHES'] = os.environ['PATCHES_BASE']
    else:
        os.environ.pop('PATCHES', None)
    E._PATCH.clear()
    return keep


if __name__ == '__main__':
    method, B, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    patches, tries, found, base = proposed(method, B)
    print(f'{method} B={B}: {tries} tries, {found} failures repaired, {len(patches)} demos patched')
    keep = validate(patches, base, out)
    json.dump(keep, open(out, 'w'), indent=1)
    print(f'kept {len(keep)} demo patches -> {out}')
    if LIB:  # the next-round library: earlier rounds + this round
        json.dump(merged(LIB, keep), open(out.replace('.json', '_lib.json'), 'w'), indent=1)
        print('library ->', out.replace('.json', '_lib.json'), len(merged(LIB, keep)), 'demos')
