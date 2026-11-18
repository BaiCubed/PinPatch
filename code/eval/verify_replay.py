"""Replay every recorded success of a run in a fresh env instance (any HTTP env) and check it still succeeds
within the step cap. Blocked actions never reached the env and are skipped.
usage: verify_replay.py ENV RUN_ID [CAP]"""
import json, sys
from run_eval import ENVS, R

envname, run = sys.argv[1], sys.argv[2]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 30
rs = [json.loads(l) for l in open(f'{R}/runs/{run}/episodes.jsonl')]
ok, bad = 0, []
for r in rs:
    if not r['success']:
        continue
    env = ENVS[envname](); env.reset(r['id']); reward = 0
    for act, obs in (r.get('trace') or [[x, ''] for x in r['actions']]):
        if obs.startswith('Blocked:'):
            continue
        o, reward, done = env.step(act)
        if o.startswith('Ambiguous request'):
            o, reward, done = env.step('0')
        if done:
            break
    env.close()
    good = env.success(reward) and r['steps'] <= cap
    ok += good
    if not good:
        bad.append((r['id'], reward))
print(json.dumps(dict(successes=sum(r['success'] for r in rs), replay_ok=ok, replay_bad=bad)))
