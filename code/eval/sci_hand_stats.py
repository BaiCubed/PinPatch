"""Main-table ScienceWorld config = hand guard + evolved library (abl/A1_shield_hand_test, 118/200).
Cost per episode, mean score (raw and clipped at 0), paired sign tests vs SFT and every same-backbone baseline.
-> runs/paper_data/sci_main_hand.csv"""
import csv
import export_paper_data as X
O = X.jl(f'{X.RUNS}/abl/A1_shield_hand_test/episodes.jsonl'); rows = []
def stats(name, D):
    v = list(D.values()); sc = [float(r.get('score', 0) or 0) for r in v]
    ids = set(O) & set(D); w = sum(bool(O[i]['success']) > bool(D[i]['success']) for i in ids); l = sum(bool(O[i]['success']) < bool(D[i]['success']) for i in ids)
    return dict(method=name, n=len(v), success=sum(bool(r['success']) for r in v), mean_score_raw=round(sum(sc) / len(sc), 1),
                mean_score_clip0=round(sum(max(0, s) for s in sc) / len(sc), 1),
                mean_llm_calls=X.mean([r.get('llm_calls', r.get('turns', 0)) or 0 for r in v]),
                mean_tokens_k=X.mean([((r.get('prompt_tokens') or 0) + (r.get('completion_tokens') or 0)) / 1000 for r in v]),
                mean_env_steps=X.mean([r.get('steps', 0) or 0 for r in v]),
                ours_better='' if name == 'ours_hand' else w, baseline_better='' if name == 'ours_hand' else l,
                sign_p='' if name == 'ours_hand' else f'{X.sign(w, l):.3g}')
rows.append(stats('ours_hand', O))
for m in ['sft', 'react', 'stateact', 'reflexion', 'adapt', 'expel', 'react_fewshot', 'react_fewshot_pp'] + ['MA-' + x for x in X.MAM]:
    try:
        rows.append(stats(m, X.load('sciworld', m)))
    except FileNotFoundError:
        pass
with open(f'{X.OUT}/sci_main_hand.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
