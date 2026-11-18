"""Dump expert demonstrations from TRAIN variations only (never dev/test). Run with the sci env python."""
import json, random
from collections import defaultdict
from scienceworld import ScienceWorldEnv
from agentenv_sciworld.environment import SciWorldEnv  # noqa
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
g = SciWorldEnv().games
sp = json.load(open(R + '/data/splits/sciworld.json'))
by = defaultdict(list)
for i in sp['train']:
    by[g[i]['taskName']].append(i)
env = ScienceWorldEnv()
rng = random.Random(20261102)
out = []
for t, v in sorted(by.items()):
    for i in rng.sample(v, min(30, len(v))):
        env.load(g[i]['taskName'], g[i]['variationIdx'], generateGoldPath=True)
        out.append(dict(id=i, task=env.getTaskDescription(), actions=env.getGoldActionSequence()))
json.dump(out, open(R + '/data/splits/sciworld_demos.json', 'w'))
print('DEMOS', len(out))
