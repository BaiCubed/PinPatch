"""Audit: how many accepted evolution repairs (and test gains) are only a flipped final binary decision.
Decision task = the task text has an 'If ...' condition naming >= 2 boxes (melting point, conductivity, thermometer).
The episode's decision = its last 'focus on <box>' / 'move ... to <box>' with a box from the condition.
Evidence before the decision: a temperature reading ('measures a temperature of') for temperature tasks; for
conductivity, an observation of the circuit's bulb/buzzer/motor state ('which is on|off') after a 'connect' action.
Classes for a successful episode: procedural (no decision in the task) | decided_with_evidence | decided_without_evidence
(a guess: the success rests on a binary choice made with no reading). Repairing episodes are re-run (deterministic)
and checked against the recorded edit outcome; round 2 uses the code frozen with it (old edit-index semantics).
  flip_audit.py accepted ROUND     ROUND in c3r1 | c3r2 | c4      -> runs/paper_data/flip_accepted_ROUND.json
  flip_audit.py test                                              -> runs/paper_data/flip_test.json"""
import json, os, re, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')); RUNS = f'{R}/runs'; OUT = f'{RUNS}/paper_data'
ROUND = sys.argv[2] if len(sys.argv) > 2 else None
if ROUND == 'c3r2':
    sys.path.insert(0, f'{R}/code/eval_frozen_c3r2')  # run_eval as frozen with the round-2 runs
sys.path.insert(1 if ROUND == 'c3r2' else 0, f'{R}/code/eval')
os.environ.setdefault('REPAIR_PORTS', '8123,8133,8124,8134,8125,8135,8126,8136')
K = lambda e: json.dumps(e, sort_keys=True)


def decision_boxes(task):
    bx = sorted({b for sent in re.split(r'\.\s', task) if sent.strip().startswith('If')
                 for b in re.findall(r'(?:the )?([a-z]+ box)', sent)})
    return bx if len(bx) >= 2 else []


def classify(task, trace):
    bx = decision_boxes(task)
    if not bx:
        return 'procedural', None
    dec = None
    for t, (a, _) in enumerate(trace):
        if any(a.endswith(b) or a.endswith(b + ' in inventory') for b in bx) and (a.startswith('focus on') or a.startswith('move')):
            dec = t
    if dec is None:
        return 'no_decision', None
    before = trace[:dec]
    if re.search(r'melting|temperature|thermometer', task):
        ev = any('measures a temperature of' in o for _, o in before)
    else:
        last_c = max([t for t, (a, _) in enumerate(before) if a.startswith('connect')], default=None)
        ev = last_c is not None and any(re.search(r'which is (on|off)', o) for _, o in before[last_c:])
    return ('decided_with_evidence' if ev else 'decided_without_evidence'), trace[dec][0]


def accepted(rnd):
    import repair, run_eval as E
    if rnd == 'c3r1':
        srcs = [(f'{RUNS}/repair', None), (f'{RUNS}/repair2', None)]; kept_of = lambda d: f'{d}/patches_gate_B2.json'; key = 'gate'
    elif rnd == 'c3r2':
        srcs = [(f'{RUNS}/repair3_gate', f'{RUNS}/repair2/patches_gate_merged.json')]; kept_of = lambda d: f'{d}/patches_gate_r2.json'; key = 'gate'
    else:
        srcs = [(f'{RUNS}/repair4', None)]; kept_of = lambda d: f'{d}/lib_gate.json'; key = 'gate'
    jobs = []
    for d, lib in srcs:
        bfiles = [f'{RUNS}/repair/base.jsonl', f'{RUNS}/repair2/base.jsonl'] if rnd == 'c4' else [f'{d}/base.jsonl']
        base = {json.loads(l)['id']: json.loads(l) for f in bfiles for l in open(f)}
        ok = {}
        for l in open(f'{d}/edits.jsonl'):
            x = json.loads(l); ok[(x['id'], K(x['edit']))] = x
        O = json.load(open(f'{d}/orders.json')); kept = json.load(open(kept_of(d)))
        if rnd == 'c4':  # lib_gate.json is the library (no earlier rounds); a patch is kept iff its demo is a key
            pass
        for i, o in O.items():
            i = int(i); hit = [e for e in o[key][:2] if ok.get((i, K(e)), {}).get('success')]
            if not hit:
                continue
            e = hit[0]; dm = str(base[i]['demo'])
            acc = dm in kept if 'demo_rank' not in e else True  # demo_rank -> exclusions on other demos; counted if any kept
            if 'demo_rank' in e:
                env = E.ENVS['sciworld'](); task = env.reset(i).split('\n')[0]; env.close()
                acc = any(str(x['id']) in kept and kept[str(x['id'])].get('exclude')
                          for x in E.retrieve(task, 1 + e['demo_rank'], exclude=i)[: e['demo_rank']])
            if acc:
                jobs.append((i, e, lib, base[i], ok[(i, K(e))]))
    print(rnd, len(jobs), 'accepted repairs', flush=True)

    def go(j):
        i, e, lib, b, rec = j
        if lib:
            os.environ['PATCHES'] = lib
        r = repair.run(i, e)
        env = E.ENVS['sciworld'](); task = env.reset(i).split('\n')[0]; env.close()
        c_rep, d_rep = classify(task, r['trace']); c_base, d_base = classify(task, b['trace'])
        return dict(id=i, edit=e, task=task, reproduced=(r['success'] == rec['success'] and r['score'] == rec['score']),
                    success=r['success'], cls=c_rep, decision=d_rep, base_cls=c_base, base_decision=d_base,
                    decision_changed=d_rep != d_base)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(go, jobs))
    json.dump(res, open(f'{OUT}/flip_accepted_{rnd}.json', 'w'), indent=1)
    good = [x for x in res if x['reproduced']]
    print(rnd, 'reproduced', len(good), '/', len(res), Counter(x['cls'] for x in good),
          'guess with changed decision', sum(x['cls'] == 'decided_without_evidence' and x['decision_changed'] for x in good))


def test():
    def jl(p):
        out = {}
        for l in open(p):
            r = json.loads(l); out.setdefault(r['id'], r)
        return out
    sys.path.insert(0, f'{R}/code/eval'); import run_eval as E
    full = jl(f'{RUNS}/frozen_c3r2/test_gate/episodes.jsonl'); lch = jl(f'{RUNS}/frozen_c3/test_lch/episodes.jsonl')
    env = E.ENVS['sciworld'](); tasks = {i: env.reset(i).split('\n')[0] for i in full}; env.close()
    gains = [i for i in full if full[i]['success'] and not lch[i]['success']]
    losses = [i for i in full if lch[i]['success'] and not full[i]['success']]
    out = dict(gains=[dict(id=i, cls=classify(tasks[i], full[i]['trace'])[0], lch_cls=classify(tasks[i], lch[i]['trace'])[0]) for i in gains],
               losses=[dict(id=i, cls=classify(tasks[i], lch[i]['trace'])[0], full_cls=classify(tasks[i], full[i]['trace'])[0]) for i in losses],
               all_full_successes=Counter(classify(tasks[i], full[i]['trace'])[0] for i in full if full[i]['success']),
               all_lch_successes=Counter(classify(tasks[i], lch[i]['trace'])[0] for i in lch if lch[i]['success']))
    json.dump(out, open(f'{OUT}/flip_test.json', 'w'), indent=1)
    print('gains', len(gains), Counter(x['cls'] for x in out['gains']))
    print('losses', len(losses), Counter(x['cls'] for x in out['losses']))
    print('all full successes', out['all_full_successes']); print('all lch successes', out['all_lch_successes'])


if __name__ == '__main__':
    test() if sys.argv[1] == 'test' else accepted(sys.argv[2])
