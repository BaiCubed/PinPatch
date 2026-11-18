"""R2(a), reviewer W4: verbalised self-diagnosis on the same procedure substrate (ScienceWorld TRAIN failures, C3 space).
The LLM reads the failed episode (task, numbered plan, executed steps with the plan step that produced each) and either
  step: names the faulty plan step -> order = skip_plan edits by |j - named step|, then the other edits (canonical order)
  rank: ranks the candidate edits directly (shown in a seeded shuffled order, so no position leak from any other order)
Outcomes of every candidate edit are already known (runs/repair/edits.jsonl), so the comparison with gate/random is offline.
  python r2a_diag.py URL NAME   -> runs/repair/orders_llm_NAME.json + runs/paper_data/r2a_NAME.csv"""
import csv, json, os, random, re, sys
from math import comb
import run_eval as E
R = E.R; D = f'{R}/runs/repair'
# R2A_V2=1 (post-hoc, PROGRESS.md 2026-11-15): phi-4 ignores the answer format and is cut at 256 tokens, so the digit
# parser read plan-step numbers as candidate ids. v2, applied identically to every diagnoser: max_tokens 1024; if the
# reply has no "Step:" / "Ranking:" marker, one follow-up turn asks for the final answer only; the ranking is parsed
# only after the last "Ranking:"; raw replies are saved. Default (unset) = v1, byte-identical behaviour.
V2 = os.environ.get('R2A_V2') == '1'
if V2:
    E.LLM['max_tokens'] = 1024
FOLLOW = dict(step='Now give only your final answer, in the form "Step: <number>".',
              rank='Now give only your final answer: "Ranking: " followed by ALL candidate numbers separated by commas.')
NF = [0]


def ask(url, q, marker, kind):
    txt, _ = E.chat(url, [dict(role='user', content=q)], 0); raw = [txt]
    if V2 and marker not in txt:
        NF[0] += 1
        txt, _ = E.chat(url, [dict(role='user', content=q), dict(role='assistant', content=txt), dict(role='user', content=FOLLOW[kind])], 0)
        raw.append(txt)
    return txt, raw


def desc(e, plan, subs):
    k, v = next(iter(e.items()))
    if k == 'skip_plan':
        return f'skip plan step {v} ("{plan[v] if v < len(plan) else "?"}")'
    if k == 'demo_rank':
        return f'use the {v + 1}-th most similar demonstration instead of the most similar one'
    return f'do not substitute "{v}" by "{subs.get(v, "?")}" when adapting the demonstration'


def episode_text(b):
    plan = '\n'.join(f'{j}. {s}' for j, s in enumerate(b['plan']))
    steps = '\n'.join(f'- [plan step {p if p >= 0 else "LLM"}] {a} -> {o[:160].replace(chr(10), " ")}'
                      for (a, o), p in zip(b['trace'][:30], b['prov'][:30]))
    return f'Plan (from a demonstration of a similar task):\n{plan}\n\nWhat was executed (and which plan step produced it):\n{steps}\n\nOutcome: FAILED (final score {b["score"]}).'


def main(url, name):
    O = json.load(open(f'{D}/orders.json')); base = {}
    for l in open(f'{D}/base.jsonl'):
        x = json.loads(l)
        if not x['edit'] and not x['success']:
            base.setdefault(str(x['id']), x)
    ok = {}
    for l in open(f'{D}/edits.jsonl'):
        x = json.loads(l); ok[(str(x['id']), json.dumps(x['edit'], sort_keys=True))] = x['success']
    out, calls = {}, 0
    for i, o in O.items():
        b = base[i]; cand = sorted(o['gate'], key=lambda e: json.dumps(e, sort_keys=True))
        env = E.ENVS['sciworld'](); obs = env.reset(int(i)); env.close(); t = obs.split('\n')[0]
        ctx = f'Task: {t}\n\n' + episode_text(b)
        q1 = ctx + '\n\nWhich single plan step is most responsible for the failure? Answer with "Step: <number>".'
        txt, raw1 = ask(url, q1, 'Step:', 'step'); calls += len(raw1)
        m = re.search(r'Step:\s*(\d+)', txt) or re.search(r'(\d+)', txt); js = int(m.group(1)) if m else 0
        skips = sorted([e for e in cand if 'skip_plan' in e], key=lambda e: (abs(e['skip_plan'] - js), e['skip_plan']))
        o_step = skips + [e for e in cand if 'skip_plan' not in e]
        sh = cand[:]; random.Random(f'{i}-r2a').shuffle(sh)
        q2 = ctx + '\n\nCandidate repairs of the procedure:\n' + '\n'.join(f'{k + 1}. {desc(e, b["plan"], b.get("subs") or {})}' for k, e in enumerate(sh)) + \
             '\n\nRank ALL candidates from most to least likely to fix the failure. Answer with "Ranking: " followed by the candidate numbers separated by commas.'
        txt2, raw2 = ask(url, q2, 'Ranking:', 'rank'); calls += len(raw2)
        part = txt2.split('Ranking:')[-1] if (not V2 or 'Ranking:' in txt2) else ''
        seq = [int(x) - 1 for x in re.findall(r'\d+', part)]
        seen = []
        for k in seq:
            if 0 <= k < len(sh) and k not in seen:
                seen.append(k)
        o_rank = [sh[k] for k in seen] + [sh[k] for k in range(len(sh)) if k not in seen]
        out[i] = dict(llm_step=o_step, llm_rank=o_rank, gate=o['gate'], named_step=js, rank_parsed=len(seen), n_cand=len(sh),
                      **(dict(raw_step=raw1, raw_rank=raw2) if V2 else {}))
    json.dump(out, open(f'{D}/orders_llm_{name}.json', 'w'))
    K = lambda i, e: ok.get((i, json.dumps(e, sort_keys=True)), False)
    rows = []
    for m in ('gate', 'llm_step', 'llm_rank'):
        firsts = [next(k + 1 for k, e in enumerate(v[m]) if K(i, e)) for i, v in out.items() if any(K(i, e) for e in v[m])]
        rec = dict(diagnoser=name, method=m, mean_first=round(sum(firsts) / len(firsts), 2))
        for B in (1, 2, 3, 4, 6, 8, 12):
            rec[f'B{B}'] = sum(any(K(i, e) for e in v[m][:B]) for i, v in out.items())
        rows.append(rec)
    rec = dict(diagnoser='-', method='random_exact')
    tot = 0.0
    for B in (1, 2, 3, 4, 6, 8, 12):
        ex = 0.0
        for i, v in out.items():
            mm = len(v['gate']); rr = sum(K(i, e) for e in v['gate']); bb = min(B, mm)
            ex += 1 - (comb(mm - rr, bb) / comb(mm, bb) if mm - rr >= bb else 0)
        rec[f'B{B}'] = round(ex, 2)
    rows.append(rec)
    print('parsed rankings', sum(v['rank_parsed'] == v['n_cand'] for v in out.values()), '/', len(out), 'calls', calls,
          'followups', NF[0], 'mean parsed frac', round(sum(v['rank_parsed'] / v['n_cand'] for v in out.values()) / len(out), 3))
    with open(f'{R}/runs/paper_data/r2a_{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
