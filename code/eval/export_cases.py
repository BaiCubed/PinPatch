"""paper_data: transitions(.csv/_score.csv) for ScienceWorld test, and case/ JSONs for 1-2 TRAIN failures repaired by the
gate's top-ranked edit (C3 full-edit data, runs/repair). Episodes are re-run (deterministic serial vLLM) with LOG_ASKS
to label each step's source; gate g_t and LCH p_t are computed on the base trace exactly as for the credit orders.
  export_cases.py transitions | cases [N]"""
import csv, json, os, sys

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')); RUNS = f'{R}/runs'; OUT = f'{RUNS}/paper_data'


def jl(path):
    out = {}
    for l in open(path):
        try:
            r = json.loads(l); out.setdefault(r['id'], r)
        except Exception:
            pass
    return out


def transitions():
    g = json.load(open(f'{R}/data/splits/sciworld_games.json'))
    cols = dict(react='r4_sci_react_test_serial', v7='frozen_sci/test_meta_replay_v7', lch='frozen_c3/test_lch',
                evo_r1='frozen_c3/test_evolved', full_r2='frozen_c3r2/test_gate', hand_only='abl/A1_shield_hand_test',
                no_constraint='abl/A1_shield_none_test')
    D = {k: jl(f'{RUNS}/{v}/episodes.jsonl') for k, v in cols.items()}
    ids = sorted(D['full_r2'])
    for name, f in (('transitions', lambda r: int(bool(r['success']))), ('transitions_score', lambda r: r['score'])):
        with open(f'{OUT}/{name}.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['task_id', 'task_type'] + list(cols))
            for i in ids:
                w.writerow([i, g[str(i)]] + [f(D[k][i]) if i in D[k] else '' for k in cols])
    print('transitions', len(ids))


def cases(n=2):
    os.environ.update(LOG_ASKS='1', REPAIR_PORTS='8123,8124')
    import numpy as np, torch
    import repair, lch, repair_eval as RE, run_eval as E
    B = jl(f'{RUNS}/repair/base.jsonl'); O = json.load(open(f'{RUNS}/repair/orders.json')); ok = {}
    for l in open(f'{RUNS}/repair/edits.jsonl'):
        x = json.loads(l); ok[(x['id'], json.dumps(x['edit'], sort_keys=True))] = x['success']
    top = [int(i) for i, o in O.items() if ok.get((int(i), json.dumps(o['gate'][0], sort_keys=True)))]
    # one per edit kind, shortest base trace first (readable figure)
    pick, kinds = ([int(x) for x in os.environ['CASE_IDS'].split(',')], {'*'}) if os.environ.get('CASE_IDS') else ([], set())
    for i in ([] if pick else sorted(top, key=lambda i: len(B[i]['trace']))):
        k = next(iter(O[str(i)]['gate'][0]))
        if k not in kinds:
            kinds.add(k); pick.append(i)
        if len(pick) >= n:
            break
    head = torch.load(f'{RUNS}/lch/v1.pt', weights_only=False); gate = RE.train_gate()
    os.makedirs(f'{OUT}/case', exist_ok=True)
    env = E.ENVS['sciworld']()
    for i in pick:
        e = O[str(i)]['gate'][0]; task = env.reset(i).split('\n')[0]

        def annotate(r, credit):
            ask_at = {}
            for a in r.get('asks') or []:
                kind = ('decide' if a['kind'].startswith('Which') else 'repair' if a['kind'].startswith('The planned')
                        else 'finish' if a['kind'].startswith('The remembered procedure') else 'llm')
                ask_at.setdefault(a['step'], kind)
            steps = []
            for t, (act, obs) in enumerate(r['trace']):
                pk = (r['prov'] or [-1] * len(r['trace']))[t]
                src = ask_at.get(t) or ('plan' if pk >= 0 else 'llm')
                row = dict(t=t, action=act, obs=obs[:200], source=src, plan_step=pk,
                           plan_text=r['plan'][pk] if pk >= 0 and r.get('plan') and pk < len(r['plan']) else None,
                           slots_in_action=[x for x, y in (r['subs'] or {}).items() if y and y in act],
                           blocked=('hand' if obs.startswith('Blocked:') and 'predicted to violate' not in obs else
                                    'learned' if 'predicted to violate' in obs else None))
                if credit:
                    row.update(gate_g=round(float(credit[0][t]), 4), lch_p=round(float(credit[1][t]), 4))
                steps.append(row)
            return steps
        def run_full(idx, ed=None):  # repair.run without dropping the ask log
            a_, sci = repair.args_for(ed); repair._tl.sci = sci
            r = E.meta_replay(idx, a_); r['subs'] = r.get('subs_used') or r.get('subs'); return r
        base = run_full(i); fixed = run_full(i, e)
        same = [a for a, _ in base['trace']] == [a for a, _ in B[i]['trace']]
        H = torch.tensor(lch.encode(RE.step_texts(base, task), (20,), bs=8)[20])
        pg = torch.sigmoid(((H - gate['mu']) / gate['sd']) @ gate['w'] + gate['b']).numpy()
        pl = torch.sigmoid(((H - head['mu']) / head['sd']) @ head['w'] + head['b']).numpy()
        out = dict(task_id=i, task=task, demo=base['demo'], plan=base['plan'], slots=base['subs'],
                   rerun_matches_logged_base=same, base_success=base['success'], base_score=base['score'],
                   gate_rank1_edit=e, gate_order_top5=O[str(i)]['gate'][:5], n_candidates=len(O[str(i)]['gate']),
                   candidates_repairing=[c for c in O[str(i)]['gate'] if ok.get((i, json.dumps(c, sort_keys=True)))],
                   base_steps=annotate(base, (pg, pl)),
                   repaired=dict(success=fixed['success'], score=fixed['score'], plan=fixed['plan'], steps=annotate(fixed, None)))
        json.dump(out, open(f'{OUT}/case/case_{i}.json', 'w'), indent=1)
        print('case', i, 'edit', e, 'base', base['success'], '->', fixed['success'], 'rerun matches', same, 'steps', len(base['trace']))
    env.close()


if __name__ == '__main__':
    transitions() if sys.argv[1] == 'transitions' else cases(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
