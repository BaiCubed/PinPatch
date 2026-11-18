"""R5: cost of credit-guided evolution (ScienceWorld, gate order, B=2, two rounds), from the run records.
Counts only what the method itself needs: base rollouts on the evolution TRAIN tasks + the first B gate-ordered edit
replays per failure (the full candidate replays in edits.jsonl were run for analysis, not by the method).
Held-out monitoring is excluded (not part of the method). -> runs/paper_data/evolution_cost.csv"""
import csv, json, os
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/runs'; B = 2; rows = []
for rnd, d in ((1, 'repair'), (2, 'repair2')):
    base = {}
    for l in open(f'{R}/{d}/base.jsonl'):
        x = json.loads(l)
        if not x['edit']:
            base.setdefault(x['id'], x)
    ed = {}
    for l in open(f'{R}/{d}/edits.jsonl'):
        x = json.loads(l); ed[(x['id'], json.dumps(x['edit'], sort_keys=True))] = x
    O = json.load(open(f'{R}/{d}/orders.json'))
    used = [ed[(int(i), json.dumps(e, sort_keys=True))] for i, o in O.items() for e in o['gate'][:B] if (int(i), json.dumps(e, sort_keys=True)) in ed]
    eps = list(base.values()) + used
    rows.append(dict(round=rnd, base_rollouts=len(base), failures=len(O), edit_rollouts=len(used), total_rollouts=len(eps),
                     env_steps=sum(x.get('steps', 0) for x in eps), llm_calls=sum(x.get('llm_calls', 0) or 0 for x in eps),
                     tokens_M=round(sum((x.get('prompt_tokens') or 0) + (x.get('completion_tokens') or 0) for x in eps) / 1e6, 2),
                     have_llm_fields=sum('llm_calls' in x for x in eps)))
tot = {k: sum(r[k] for r in rows) for k in rows[0] if k != 'round'}; tot['round'] = 'total'; rows.append(tot)
with open(f'{R}/paper_data/evolution_cost.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
