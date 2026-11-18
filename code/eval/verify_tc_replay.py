"""Independent check of a frozen TextCraft test run, without touching frozen files:
re-run the frozen arm with action logging attached from outside (deterministic, so it must give the same
outcomes), then replay every recorded success step by step in a brand-new env instance.
usage: PYTHONHASHSEED=0 verify_tc_replay.py WORKDIR ARM SPLIT"""
import json, random, sys, pickle
wd, arm, split = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ['ag_run.py', wd, 'noop']
import ag_run
from ag_run import AGEpisode
from pinpatch_tc import run
from run_eval import TextCraft

logs = {}


class Logged(AGEpisode):
    def __init__(self, inst, *a, **k):
        super().__init__(inst, *a, **k); logs[inst['id']] = self


run.Episode = Logged
model = pickle.load(open(wd + '/model.pkl', 'rb'))
rng = random.Random(0)  # evaluate() uses Random(seed=0) over the split in order
res = [run.run_episode(x, model, rng, learn=False, **run.ARMS[arm]) for x in run.load(split)]
frozen = json.load(open(f'{wd}/results/{split}_{arm}.json'))
same = [r['won'] for r in res] == [e['won'] for e in frozen['episodes']]
ok = bad = 0
for r in res:
    if not r['won']:
        continue
    env = TextCraft(); env.reset(r['id']); won = False
    for i, (act, _) in enumerate(logs[r['id']].log):
        obs, rew, done = env.step(act)
        won = won or rew > 0
        if done:
            break
    ok += won and i + 1 <= 40; bad += not (won and i + 1 <= 40)
print(json.dumps(dict(rerun_matches_frozen=same, successes=sum(r['won'] for r in res), replay_ok=ok, replay_bad=bad)))
