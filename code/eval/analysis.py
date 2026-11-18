"""Offline analyses (pre-registered in PROGRESS.md, section C and B4/B6): no new episodes, existing runs only.
  analysis.py cost | types | credit | patches | failures | lch | all"""
import json, os, re, sys
from collections import Counter, defaultdict
from math import comb

import numpy as np

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
RUNS = f'{R}/runs'
sys.path.insert(0, f'{R}/code/eval')


def load(path):
    out = {}
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); out.setdefault(r['id'], r)
            except Exception:
                pass
    return out


def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)


OURS = dict(alfworld='frozen_alf/test_meta_alf', sciworld='frozen_c3r2/test_gate', webshop='frozen_ws/test_meta_ws')
BASE = dict(alfworld=dict(react='r4_alf_react_test_serial'), sciworld=dict(react='r4_sci_react_test_serial'),
            webshop=dict(react='b1_ws_react_test_serial'), textcraft=dict(react='r1_tc_react_test_serial'))
SHORT = dict(textcraft='tc', alfworld='alf', sciworld='sci', webshop='ws')
for e in BASE:
    for m in ('stateact', 'reflexion', 'adapt', 'expel'):
        BASE[e][m] = f'b1_{SHORT[e]}_{m}_test_serial'


def cost():
    print('\n### C1 cost per episode (test): env steps, LLM calls, prompt+completion tokens (k)')
    print('| env | method | success | steps | LLM calls | tokens (k) |\n|---|---|---|---|---|---|')
    for env in ('alfworld', 'sciworld', 'webshop', 'textcraft'):
        runs = ([('PinPatch (ours)', OURS[env])] if env in OURS else []) + list(BASE[env].items())
        for name, run in runs:
            M = load(f'{RUNS}/{run}/episodes.jsonl')
            if not M:
                continue
            v = list(M.values()); f = lambda k: np.mean([r.get(k, 0) or 0 for r in v])
            tok = np.mean([(r.get('prompt_tokens', 0) or 0) + (r.get('completion_tokens', 0) or 0) for r in v]) / 1000
            succ = sum(bool(r['success']) for r in v)
            print(f'| {env} | {name} | {succ}/{len(v)} | {f("steps"):.1f} | {f("llm_calls"):.1f} | {tok:.1f} |')


def types():
    g = json.load(open(f'{R}/data/splits/sciworld_games.json'))
    cols = dict(v7='frozen_sci/test_meta_replay_v7', lch='frozen_c3/test_lch', r1='frozen_c3/test_evolved',
                full='frozen_c3r2/test_gate', react='r4_sci_react_test_serial')
    D = {k: load(f'{RUNS}/{v}/episodes.jsonl') for k, v in cols.items()}
    by = defaultdict(list)
    for i in D['full']:
        by[g[str(i)]].append(i)
    print('\n### C2 ScienceWorld test success by task type')
    print('| type | n | ' + ' | '.join(cols) + ' |\n|' + '---|' * (len(cols) + 2))
    for t in sorted(by):
        ids = by[t]
        print(f'| {t} | {len(ids)} | ' + ' | '.join(str(sum(D[k].get(i, {}).get('success', False) for i in ids)) for k in cols) + ' |')


def credit():
    """B4 (C3 small space, every candidate run) + C6: search cost and localization of credit orders."""
    from repair import candidates  # noqa (imports run_eval; only used for consistency)
    for d in ('repair',):
        O = json.load(open(f'{RUNS}/{d}/orders.json')); ok = {}
        for l in open(f'{RUNS}/{d}/edits.jsonl'):
            x = json.loads(l); ok[(x['id'], json.dumps(x['edit'], sort_keys=True))] = x['success']
        K = lambda i, e: ok.get((int(i), json.dumps(e, sort_keys=True)), False)
        n = len(O); fixable = [i for i, o in O.items() if any(K(i, e) for e in o['gate'])]
        print(f'\n### B4 search budget, C3 space ({d}: {n} failures, {len(fixable)} repairable by some edit)')
        print('| B | gate | lch | recency | random (exact E) | oracle |\n|---|---|---|---|---|---|')
        for B in (1, 2, 3, 4, 6, 8, 12, 99):
            row = []
            for m in ('gate', 'lch', 'recency'):
                row.append(sum(any(K(i, e) for e in o[m][:B]) for i, o in O.items()))
            ex = 0.0  # random order: P(at least one repair in the first B of m candidates with r repairs)
            for i, o in O.items():
                m_, r_ = len(o['gate']), sum(K(i, e) for e in o['gate']); b = min(B, m_)
                ex += 1 - (comb(m_ - r_, b) / comb(m_, b) if m_ - r_ >= b else 0)
            row += [round(ex, 1), sum(1 for i in fixable)]
            print(f'| {"all" if B == 99 else B} | ' + ' | '.join(str(x) for x in row) + ' |')
        # tries to first repair on repairable failures
        print('\nmean tries to first repair (repairable failures):', end=' ')
        for m in ('gate', 'lch', 'recency'):
            t = [next(k + 1 for k, e in enumerate(O[i][m]) if K(i, e)) for i in fixable]
            print(f'{m} {np.mean(t):.2f}', end='; ')
        ex = [(len(O[i]['gate']) + 1) / (sum(K(i, e) for e in O[i]['gate']) + 1) for i in fixable]
        print(f'random {np.mean(ex):.2f}')
        print('\n### C6 credit localization: P(a repairing edit is ranked above a non-repairing one), per failure, mean')
        for m in ('gate', 'lch', 'recency', 'random'):
            au = []
            for i in fixable:
                s = [K(i, e) for e in O[i][m]]; pos = [k for k, x in enumerate(s) if x]; neg = [k for k, x in enumerate(s) if not x]
                if pos and neg:
                    au.append(np.mean([p < q for p in pos for q in neg]))
            print(f'{m}: AUC {np.mean(au):.3f} over {len(au)} failures')
    # C4 large space (B=1,2 always; 4 when the B=4 extension has run)
    D4 = f'{RUNS}/repair4'
    if os.path.exists(f'{D4}/orders.json'):
        O = json.load(open(f'{D4}/orders.json')); ok = {}
        for l in open(f'{D4}/edits.jsonl'):
            x = json.loads(l); ok[(x['id'], json.dumps(x['edit'], sort_keys=True))] = x['success']
        print('\n### B4 search budget, C4 space (799 failures, ~608 candidates each): failures repaired')
        print('| B | gate | random | simple | runs missing |\n|---|---|---|---|---|')
        for B in (1, 2, 3, 4):
            row, miss = [], 0
            for m in ('gate', 'random', 'simple'):
                if m == 'simple' and B > 2:
                    row.append('–'); continue
                miss += sum((int(i), json.dumps(e, sort_keys=True)) not in ok for i, o in O.items() for e in o[m][:B])
                row.append(sum(any(ok.get((int(i), json.dumps(e, sort_keys=True))) for e in o[m][:B]) for i, o in O.items()))
            print(f'| {B} | ' + ' | '.join(str(x) for x in row) + f' | {miss} |')


def patches():
    import run_eval as E
    demos = {str(d['id']): d for d in json.load(open(f'{R}/data/splits/sciworld_demos.json'))}
    print('\n### C4 what evolution changed (libraries)')
    for name, f in [('r1 gate (validated)', f'{RUNS}/repair2/patches_gate_merged.json'),
                    ('r2 gate (validated, full system)', f'{R}/code/eval_frozen_c3r2/lib_gate_r2.json'),
                    ('r1 gate (no validation)', f'{RUNS}/abl/lib_r1_gate_noval.json')]:
        if not os.path.exists(f):
            continue
        L = json.load(open(f)); c = Counter(); verbs = Counter()
        for d, p in L.items():
            c['exclude'] += bool(p.get('exclude')); c['drop_sub'] += len(p.get('drop_sub', [])); c['skip'] += len(p.get('skip_plan', []))
            plan = E.expand_loops(E.compress(demos[d]['actions'])) if d in demos else []
            for k in p.get('skip_plan', []):
                if k < len(plan):
                    s = plan[k]; verbs['loop' if s.startswith('<loop') else s.split()[0]] += 1
        print(f'{name}: {len(L)} demos; excluded demos {c["exclude"]}, dropped slot bindings {c["drop_sub"]}, '
              f'skipped steps {c["skip"]}; skipped-step verbs {dict(verbs.most_common(8))}')


def failures():
    print('\n### C5 failure taxonomy, ScienceWorld test (failed episodes)')
    cols = dict(v7='frozen_sci/test_meta_replay_v7', lch='frozen_c3/test_lch', full='frozen_c3r2/test_gate',
                react='r4_sci_react_test_serial')
    print('| run | failed | step cap | ended early (task failed) | stopped <cap, not done | partial score>0 |\n|---|---|---|---|---|---|')
    for k, run in cols.items():
        M = load(f'{RUNS}/{run}/episodes.jsonl'); F = [r for r in M.values() if not r['success']]
        cap = sum(r['steps'] >= 30 for r in F)
        early = sum(r['steps'] < 30 and (r.get('score', 0) or 0) < 0 for r in F)
        other = len(F) - cap - early
        part = sum((r.get('score', 0) or 0) > 0 for r in F)
        print(f'| {k} | {len(F)} | {cap} | {early} | {other} | {part} |')


def lch():
    import torch
    h = torch.load(f'{RUNS}/lch/v1.pt', weights_only=False)
    print('\n### B6 LCH layer (held-out 20% of TRAIN episodes; tau picked per layer by recall - 3*false_block)')
    print(json.dumps(h['report']), '| chosen layer', h['layer'], 'tau', h['tau'], 'n', h['n'], 'pos', h['n_pos'])
    d = np.load(f'{RUNS}/lch/v1.npz'); y = d['y']; ep = d['ep']
    import lch as L
    import random
    eps = sorted(set(ep.tolist())); random.Random(20261106).shuffle(eps); hold = set(eps[: len(eps) // 5])
    te = np.array([e in hold for e in ep])
    p = L.prob(h, torch.tensor(d[f'h{h["layer"]}'][te])); p = np.asarray(p); yt = y[te]
    print('\n### B1 (offline) threshold vs held-out recall / false-block')
    for t in (0.5, 0.8, 0.9, 0.95, 0.9761, 0.99, 0.995):
        print(f'tau {t}: recall {((p > t) & (yt == 1)).sum() / max(1, (yt == 1).sum()):.3f}  '
              f'false_block {((p > t) & (yt == 0)).sum() / max(1, (yt == 0).sum()):.4f}')


if __name__ == '__main__':
    fs = dict(cost=cost, types=types, credit=credit, patches=patches, failures=failures, lch=lch)
    for k in (fs if sys.argv[1] == 'all' else sys.argv[1:]):
        try:
            fs[k]()
        except Exception as ex:
            print(f'[{k} failed: {type(ex).__name__}: {ex}]')
