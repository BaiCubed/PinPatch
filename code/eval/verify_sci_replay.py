"""Replay every recorded ScienceWorld success in a fresh env instance and check it still scores 100
within the step cap. Blocked actions never reached the env, so they are skipped (but they did count as
steps in the run). usage: verify_sci_replay.py RUN_ID [CAP]"""
import json, sys
from run_eval import SciWorld, R

run, cap = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30
rs = [json.loads(l) for l in open(f'{R}/runs/{run}/episodes.jsonl')]
ok, bad = 0, []
for r in rs:
    if not r['success']:
        continue
    env = SciWorld(); env.reset(r['id']); score = 0
    for act, obs in r['trace']:
        if obs.startswith('Blocked:'):
            continue
        o, score, done = env.step(act)
        if o.startswith('Ambiguous request'):
            o, score, done = env.step('0')
        if done:
            break
    env.close()
    good = score >= 100 and r['steps'] <= cap
    ok += good
    if not good:
        bad.append((r['id'], score))
print(json.dumps(dict(successes=sum(r['success'] for r in rs), replay_ok=ok, replay_bad=bad)))
