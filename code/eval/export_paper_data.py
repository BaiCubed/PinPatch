"""Export per-method data for the paper's figures -> runs/paper_data/*.csv (copied to the repo's paper_data/).
cost.csv, types.csv, budget.csv, failures.csv, m2_pvalues.csv. Test splits only; same loaders as compare_all.py."""
import csv, json, os
from math import comb

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')); RUNS = f'{R}/runs'; MA = f'{R}/code/MetaAct/logs/runs'
OUT = f'{RUNS}/paper_data'; os.makedirs(OUT, exist_ok=True)
ENVS = ('textcraft', 'alfworld', 'sciworld', 'webshop'); SHORT = dict(textcraft='tc', alfworld='alf', sciworld='sci', webshop='ws')
OURS = dict(alfworld='frozen_alf/test_meta_alf', sciworld='frozen_c3r2/test_gate', webshop='frozen_ws/test_meta_ws')
REACT = dict(textcraft='r1_tc_react_test_serial', alfworld='r4_alf_react_test_serial', sciworld='r4_sci_react_test_serial',
             webshop='b1_ws_react_test_serial')
MAM = ['metaact', 'ace', 'dc', 'reflact', 'rpms', 'skillrl', 'agentspec', 'ontoguard']


def jl(path):
    out = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        out.setdefault(r['id'], r)
    return out


def load(env, method):
    if method == 'ours':
        if env == 'textcraft':
            d = json.load(open(f'{RUNS}/phase2_ag_base/merged/results/test_mem_analog_gated.json'))
            return {x['id']: dict(success=x['won'], score=float(x['won']), steps=x['steps']) for x in d['episodes']}
        return jl(f'{RUNS}/{OURS[env]}/episodes.jsonl')
    if method == 'react_fewshot':  # R1: demo-augmented ReAct (AgentTraj-L TRAIN exemplars)
        return jl(f'{RUNS}/r1_fewshot/{env}_test/episodes.jsonl')
    if method == 'react_fewshot_pp':  # R1: same, exemplars from PinPatch's own demo pool (ScienceWorld only)
        return jl(f'{RUNS}/r1_fewshot/{env}_pp_test/episodes.jsonl')
    if method == 'MA-reflact_tolerant':  # R3 appendix: ReflAct with the tolerant action parse
        method = 'MA-reflact'; env_sfx = '_tolerant'
    else:
        env_sfx = ''
    if method.startswith('MA-'):
        out = {}
        for i, r in jl(f'{MA}/ma_{method[3:]}_{env}_test{env_sfx}/results.jsonl').items():
            r = dict(r); r['score'] = r.get('score', 0) / (100.0 if env == 'webshop' else 1)
            out[int(i.split('_')[1])] = r
        return out
    if method == 'sft':  # P0-3: LoRA SFT on AgentTraj-L (test/dev overlaps removed), AgentTraj prompts
        return jl(f'{RUNS}/p03_sft/{env}_sft_test/episodes.jsonl')
    if method.startswith('phi4-'):  # P0-2 second backbone (microsoft/phi-4), ScienceWorld and ALFWorld only
        return jl(f'{RUNS}/p02_phi4/{SHORT[env]}_{method[5:]}_test/episodes.jsonl')
    run = REACT[env] if method == 'react' else f'b1_{SHORT[env]}_{method}_test_serial'
    return jl(f'{RUNS}/{run}/episodes.jsonl')


def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)


def mean(v):
    return round(sum(v) / len(v), 2) if v else ''


# 1 cost.csv (+ 5 m2_pvalues.csv)
cost, pv = [], []
for env in ENVS:
    O = load(env, 'ours'); key = 'score' if env == 'webshop' else 'success'
    for m in ['ours', 'react', 'stateact', 'reflexion', 'adapt', 'expel', 'sft'] + ['MA-' + x for x in MAM] + \
             ['react_fewshot', 'react_fewshot_pp', 'MA-reflact_tolerant']:
        try:
            D = load(env, m)
        except FileNotFoundError:
            continue
        v = list(D.values()); ma = m.startswith('MA-')
        has_calls = any('llm_calls' in r for r in v)
        calls = mean([r.get('llm_calls', 0) or 0 for r in v]) if has_calls else (mean([r.get('turns', 0) for r in v]) if ma else '')
        has_tok = any('prompt_tokens' in r for r in v)
        tok = mean([((r.get('prompt_tokens') or 0) + (r.get('completion_tokens') or 0)) / 1000 for r in v]) if has_tok else ''
        note = ('llm_calls = agent turns (one LLM call per turn; excludes offline synthesis/reflection calls)' if ma and not has_calls
                else 'TextCraft pipeline logs no LLM calls/tokens' if env == 'textcraft' and m == 'ours' else '')
        cost.append(dict(env=env, method=m, success=sum(bool(r['success']) for r in v), n=len(v),
                         mean_reward=mean([100 * float(r.get('score', 0) or 0) for r in v]) if env == 'webshop' else '',
                         mean_env_steps=mean([r.get('steps', 0) or 0 for r in v]), mean_llm_calls=calls, mean_tokens_k=tok, note=note))
        if m != 'ours':
            ids = set(O) & set(D); val = lambda r: float(r[key])
            w = sum(val(O[i]) > val(D[i]) for i in ids); l = sum(val(O[i]) < val(D[i]) for i in ids)
            pv.append(dict(env=env, method=m, metric=key, n_paired=len(ids), ours=round(sum(val(O[i]) for i in ids), 2),
                           baseline=round(sum(val(D[i]) for i in ids), 2), ours_better=w, baseline_better=l, sign_p=f'{sign(w, l):.3g}'))

# 1b backbone.csv: other backbones (phi-4 = P0-2; granite8b / mistral7b / olmo3_7b = P0-2b); paired tests are against
# ours on the SAME backbone. ScienceWorld reference = main-table config (hand guard + evolved library, 'ourshand') when it
# exists, else 'ours' (hand+v1). *_norm rows = R9 action normalisation (appendix only).
BACKBONES = dict(phi4=('phi-4', 'p02_phi4'), granite8b=('Granite-3.3-8B', 'p04_family/granite8b'),
                 mistral7b=('Mistral-7B-v0.3', 'p04_family/mistral7b'), olmo3_7b=('Olmo-3-7B', 'p04_family/olmo3_7b'))
bb = []
for key, (bname, root) in BACKBONES.items():
    for env in ('sciworld', 'alfworld'):
        s_ = SHORT[env]; P = lambda m: f'{RUNS}/{root}/{s_}_{m}_test/episodes.jsonl'
        ref = 'ourshand' if os.path.exists(P('ourshand')) else 'ours'
        if not os.path.exists(P(ref)):
            continue
        O = jl(P(ref))
        for m in ('ourshand', 'ours', 'react', 'reflexion', 'react_norm', 'reflexion_norm'):
            if not os.path.exists(P(m)):
                continue
            D = jl(P(m)); ids = set(O) & set(D)
            w = sum(bool(O[i]['success']) > bool(D[i]['success']) for i in ids); l = sum(bool(O[i]['success']) < bool(D[i]['success']) for i in ids)
            bb.append(dict(backbone=bname, env=env, method=m, success=sum(bool(r['success']) for r in D.values()), n=len(D),
                           mean_llm_calls=mean([r.get('llm_calls', 0) or 0 for r in D.values()]),
                           mean_tokens_k=mean([((r.get('prompt_tokens') or 0) + (r.get('completion_tokens') or 0)) / 1000 for r in D.values()]),
                           mean_score=mean([float(r.get('score', 0) or 0) for r in D.values()]),
                           ours_better=w if m != ref else '', baseline_better=l if m != ref else '',
                           sign_p=f'{sign(w, l):.3g}' if m != ref else '', complete=len(D) == 200, paired_ref=ref))

# 2 types.csv
g = json.load(open(f'{R}/data/splits/sciworld_games.json'))
cols = dict(v7='frozen_sci/test_meta_replay_v7', lch='frozen_c3/test_lch', evo_r1='frozen_c3/test_evolved',
            full='frozen_c3r2/test_gate', react='r4_sci_react_test_serial')
T = {k: jl(f'{RUNS}/{v}/episodes.jsonl') for k, v in cols.items()}; by = {}
for i in T['full']:
    by.setdefault(g[str(i)], []).append(i)
types = [dict(task_type=t, n=len(ids), **{k: sum(T[k].get(i, {}).get('success', False) for i in ids) for k in cols})
         for t, ids in sorted(by.items(), key=lambda x: -len(x[1]))]

# 3 budget.csv
budget = []
O3 = json.load(open(f'{RUNS}/repair/orders.json')); ok = {}
for l in open(f'{RUNS}/repair/edits.jsonl'):
    x = json.loads(l); ok[(x['id'], json.dumps(x['edit'], sort_keys=True))] = x['success']
K = lambda d, i, e: d.get((int(i), json.dumps(e, sort_keys=True)), False)
for B in range(1, 13):
    for m in ('gate', 'lch', 'recency'):
        budget.append(dict(space='c3', method=m, B=B, repaired=sum(any(K(ok, i, e) for e in o[m][:B]) for i, o in O3.items()), n_fail=len(O3), note=''))
    ex = 0.0
    for i, o in O3.items():
        mm, rr = len(o['gate']), sum(K(ok, i, e) for e in o['gate']); b = min(B, mm)
        ex += 1 - (comb(mm - rr, b) / comb(mm, b) if mm - rr >= b else 0)
    budget.append(dict(space='c3', method='random', B=B, repaired=round(ex, 2), n_fail=len(O3), note='exact expectation over random orders'))
    budget.append(dict(space='c3', method='oracle', B=B, repaired=sum(any(K(ok, i, e) for e in o['gate']) for i, o in O3.items()), n_fail=len(O3), note='any repairing edit exists'))
O4 = json.load(open(f'{RUNS}/repair4/orders.json')); ok4 = {}
for l in open(f'{RUNS}/repair4/edits.jsonl'):
    x = json.loads(l); ok4[(x['id'], json.dumps(x['edit'], sort_keys=True))] = x['success']
for B in (1, 2, 3, 4):
    for m in ('gate', 'random', 'simple'):
        if m == 'simple' and B > 2:
            continue
        need = [(int(i), json.dumps(e, sort_keys=True)) for i, o in O4.items() for e in o[m][:B]]
        if any(k not in ok4 for k in need):
            continue  # this budget not fully run yet
        budget.append(dict(space='c4', method=m, B=B, repaired=sum(any(K(ok4, i, e) for e in o[m][:B]) for i, o in O4.items()),
                           n_fail=len(O4), note='random = one seeded random order per failure'))

# 4 failures.csv
fails = []
for k, run in dict(react='r4_sci_react_test_serial', v7='frozen_sci/test_meta_replay_v7', lch='frozen_c3/test_lch',
                   full='frozen_c3r2/test_gate').items():
    F = [r for r in jl(f'{RUNS}/{run}/episodes.jsonl').values() if not r['success']]
    cap = sum(r['steps'] >= 30 for r in F); early = sum(r['steps'] < 30 and (r.get('score') or 0) < 0 for r in F)
    fails.append(dict(system=k, n_fail=len(F), step_cap=cap, early_fail=early, other=len(F) - cap - early,
                      partial_score_gt0=sum((r.get('score') or 0) > 0 for r in F)))

for name, rows in dict(cost=cost, backbone=bb, types=types, budget=budget, failures=fails, m2_pvalues=pv).items():
    with open(f'{OUT}/{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(name, len(rows))
