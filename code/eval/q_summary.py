"""Queue summary (runs after every queue job): raw, unreviewed numbers -> runs/paper_data/pending/*.csv + STATUS.md.
Everything here is [pending] until reviewed. Importing export_paper_data also refreshes the main CSVs (cost, backbone, ...).
Paired tests are sign tests on per-episode success; 'neg' = episodes with a negative score (ScienceWorld catastrophic)."""
import csv, glob, json, os, statistics as st
import export_paper_data as X

RUNS = X.RUNS; OUT = f'{RUNS}/paper_data/pending'; os.makedirs(OUT, exist_ok=True)


def L(rid):
    p = f'{RUNS}/{rid}/episodes.jsonl'
    return X.jl(p) if os.path.exists(p) else None


def row(name, rid, ref=None, refname='', n_full=200, **extra):
    D = L(rid)
    if D is None:
        return dict(config=name, run=rid, n=0, **extra)
    sc = [float(r.get('score', 0) or 0) for r in D.values()]
    out = dict(config=name, run=rid, n=len(D), complete=len(D) >= n_full, success=sum(bool(r['success']) for r in D.values()),
               mean_score=round(sum(sc) / len(sc), 1), neg=sum(s < 0 for s in sc),
               mean_llm_calls=X.mean([r.get('llm_calls', 0) or 0 for r in D.values()]),
               mean_tokens_k=X.mean([((r.get('prompt_tokens') or 0) + (r.get('completion_tokens') or 0)) / 1000 for r in D.values()]), **extra)
    R_ = L(ref) if ref else None
    if R_:
        ids = set(D) & set(R_); w = sum(bool(D[i]['success']) > bool(R_[i]['success']) for i in ids)
        l = sum(bool(D[i]['success']) < bool(R_[i]['success']) for i in ids)
        out.update(vs=refname, better=w, worse=l, sign_p=f'{X.sign(w, l):.3g}')
    return out


def dump(name, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(f'{OUT}/{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


HAND, HANDV1, V1, NONE = 'abl/A1_shield_hand_test', 'frozen_c3r2/test_gate', 'abl/A1_shield_learned_test', 'abl/A1_shield_none_test'
# R4 non-hidden-state constraint baselines
ho = json.load(open(f'{RUNS}/r4/heldout_report.json')) if os.path.exists(f'{RUNS}/r4/heldout_report.json') else {}
dump('r4_constraint_text', [
    row('none', NONE), row('hand', HAND), row('v1 hidden-state only', V1, **ho.get('v1', {})), row('hand+v1 hidden-state', HANDV1),
    row('lookup only', 'r4/test_lookup_learned', V1, 'v1 only', **ho.get('lookup', {})), row('hand+lookup', 'r4/test_lookup_both', HANDV1, 'hand+v1'),
    row('tfidf only', 'r4/test_tfidf_learned', V1, 'v1 only', **ho.get('tfidf', {})), row('hand+tfidf', 'r4/test_tfidf_both', HANDV1, 'hand+v1')])
# R7 seeds at T=0.7
r7 = [row('hand+lib T=0', HAND), row('hand nolib T=0', 'r3/test_hand_nolib', HAND, 'hand+lib T=0')]
for s in (1, 2, 3):
    r7.append(row(f'hand+lib T=0.7 s{s}', f'r7/test_hand_lib_T07_s{s}'))
    r7.append(row(f'hand nolib T=0.7 s{s}', f'r7/test_hand_nolib_T07_s{s}', f'r7/test_hand_lib_T07_s{s}', f'hand+lib T=0.7 s{s}'))
for k in ('hand+lib', 'hand nolib'):
    v = [r['success'] for r in r7 if r['config'].startswith(k + ' T=0.7') and r.get('complete')]
    if len(v) >= 2:
        r7.append(dict(config=f'{k} T=0.7 mean (sd)', n=len(v), success=round(st.mean(v), 1), sd=round(st.stdev(v), 1)))
dump('r7_seeds', r7)
# R8 SFT as the decision-point LLM
dump('r8_sft_decision', [row('PinPatch main (Qwen decisions)', HAND), row('SFT ReAct', 'p03_sft/sciworld_sft_test'),
                         row('PinPatch, SFT decisions', 'r8/test_sft_decision', HAND, 'PinPatch main'),
                         row('PinPatch, SFT decisions vs SFT', 'r8/test_sft_decision', 'p03_sft/sciworld_sft_test', 'SFT ReAct')])
# R2(b) evolution with the LLM diagnoser's order
nho = len(json.load(open(f'{X.R}/data/splits/sciworld_train_ho2all.json')))
r2b = []
for lib in ('base', 'gate', 'random', 'llm_qwen', 'llm_phi4'):
    pp = f'{RUNS}/r2b/patches_{lib}_B2.json'
    r2b.append(row(f'held-out {lib}', f'r2b/ho_{lib}', None if lib == 'gate' else 'r2b/ho_gate', 'gate', n_full=nho,
                   demos_patched=len(json.load(open(pp))) if os.path.exists(pp) else ''))
for lib in ('gate', 'llm_qwen', 'llm_phi4'):
    if L(f'r2b/test_{lib}'):
        r2b.append(row(f'test {lib}', f'r2b/test_{lib}', None if lib == 'gate' else 'r2b/test_gate', 'gate'))
dump('r2b_evolution', r2b)
# R2(a) all diagnosers in one file
ra = []
for p in sorted(glob.glob(f'{RUNS}/paper_data/r2a_*.csv')):
    ra += [dict(file=os.path.basename(p), **r) for r in csv.DictReader(open(p))]
dump('r2a_all', ra)
# status of every queued / still-running run
watch = ['r1_fewshot/textcraft_test', 'r1_fewshot/alfworld_test', 'r1_fewshot/sciworld_test', 'r1_fewshot/webshop_test',
         'r1_fewshot/sciworld_pp_test', 'r4/test_lookup_learned', 'r4/test_lookup_both', 'r4/test_tfidf_learned',
         'r4/test_tfidf_both', 'r8/test_sft_decision'] + [f'r7/test_hand_{k}_T07_s{s}' for k in ('lib', 'nolib') for s in (1, 2, 3)] + \
        [f'p04_family/{f}/{e}_{m}_test' for f in ('granite8b', 'mistral7b', 'olmo3_7b') for e, m in
         (('sci', 'ourshand'), ('sci', 'react'), ('sci', 'reflexion'), ('alf', 'react'), ('alf', 'reflexion'))] + \
        [f'p02_phi4/{e}_{m}_norm_test' for e in ('alf', 'sci') for m in ('react', 'reflexion')] + \
        [f'r2b/ho_{l}' for l in ('base', 'gate', 'random', 'llm_qwen', 'llm_phi4')]
q = f'{RUNS}/queue/status'
with open(f'{OUT}/STATUS.md', 'w') as f:
    f.write('# Queue status (raw, pending review)\n\n| run | episodes |\n|---|---|\n')
    for rid in watch:
        f.write(f'| {rid} | {(len(L(rid)) if L(rid) else 0)} |\n')
    f.write('\n| job | state |\n|---|---|\n')
    for p in sorted(glob.glob(f'{X.R}/runs/setup/queue/jobs/[0-9]*.sh')):
        n = os.path.basename(p)[:-3]
        s = 'DONE' if os.path.exists(f'{q}/{n}.done') else 'FAIL' if os.path.exists(f'{q}/{n}.fail') else \
            'running' if os.path.isdir(f'{X.R}/runs/queue/claim/{n}') else 'waiting'
        f.write(f'| {n} | {s} |\n')
print('pending summary ->', OUT)
