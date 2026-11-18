"""Final table: frozen PinPatch vs every same-backbone baseline on the four AgentGym test splits, with paired sign
tests (success for TextCraft/ALFWorld/ScienceWorld, reward for WebShop). Duplicate lines per id: first one counts."""
import json
from math import comb
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/runs'
MA = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/code/MetaAct/logs/runs'
MA_METHODS = ['metaact', 'ace', 'dc', 'reflact', 'rpms', 'skillrl', 'agentspec', 'ontoguard']
META = dict(textcraft=None, alfworld='frozen_alf/test_meta_alf', sciworld='frozen_sci/test_meta_replay_v7',
            webshop='frozen_ws/test_meta_ws')
SHORT = dict(textcraft='tc', alfworld='alf', sciworld='sci', webshop='ws')
BASE = dict(textcraft=dict(react='r1_tc_react_test_serial'), alfworld=dict(react='r4_alf_react_test_serial'),
            sciworld=dict(react='r4_sci_react_test_serial', react_demo='r3_sci_react_demo_test_serial'),
            webshop=dict(react='b1_ws_react_test_serial'))
for e in BASE:
    for m in ('stateact', 'reflexion', 'adapt', 'expel'):
        BASE[e][m] = f'b1_{SHORT[e]}_{m}_test_serial'
    for m in MA_METHODS:
        BASE[e]['MA-' + m] = f'ma:ma_{m}_{e}_test'




def load(run):
    out = {}
    if run.startswith('ma:'):  # MetaAct-repo method on our env (metaact_ag.py): ids 'env_idx', WebShop score 0-100
        for l in open(f'{MA}/{run[3:]}/results.jsonl'):
            r = json.loads(l); i = int(r['id'].split('_')[1])
            if 'webshop' in run:
                r['score'] = r.get('score', 0) / 100.0
            out.setdefault(i, r)
        return out
    for l in open(f'{R}/{run}/episodes.jsonl'):
        r = json.loads(l); out.setdefault(r['id'], r)
    return out


def meta(env):
    if env == 'textcraft':
        d = json.load(open(f'{R}/phase2_ag_base/merged/results/test_mem_analog_gated.json'))
        return {x['id']: dict(success=x['won'], score=float(x['won'])) for x in d['episodes']}
    return load(META[env])


def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(w, l) + 1)) / 2 ** n)


rows = []
for env in ('textcraft', 'alfworld', 'sciworld', 'webshop'):
    M = meta(env); key = 'score' if env == 'webshop' else 'success'
    val = lambda r: float(r[key])
    msc = 100 * sum(val(r) for r in M.values()) / len(M)
    print(f'\n== {env} (n={len(M)})  PinPatch: success {sum(r["success"] for r in M.values())}'
          + (f', mean reward {msc:.1f}' if env == 'webshop' else ''))
    for name, run in BASE[env].items():
        try:
            B = load(run)
        except FileNotFoundError:
            print(f'   {name:11s} missing'); continue
        ids = [i for i in M if i in B]
        w = sum(val(M[i]) > val(B[i]) for i in ids); l = sum(val(M[i]) < val(B[i]) for i in ids)
        bs = sum(B[i]['success'] for i in ids); bm = 100 * sum(float(B[i].get('score', 0)) for i in ids) / max(1, len(ids))
        tag = '' if len(ids) == len(M) else f'  (partial {len(ids)}/{len(M)})'
        print(f'   {name:11s} success {bs:3d}' + (f'  mean reward {bm:5.1f}' if env == 'webshop' else '')
              + f'   PinPatch wins {w:3d} / losses {l:3d}   sign p={sign(w, l):.1e}{tag}')
        rows.append(dict(env=env, method=name, n=len(ids), base_success=bs, base_mean=bm, wins=w, losses=l, p=sign(w, l)))
json.dump(rows, open(f'{R}/compare_all.json', 'w'), indent=1)
