"""ScienceWorld dev 60 from non-test variations of the public test task types; train = the rest.
Only the public task-type list of the test file is used, no results. Run with the sci env python."""
import json, random
from collections import Counter
from agentenv_sciworld.environment import SciWorldEnv  # noqa
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
g = SciWorldEnv().games
test = [int(x['item_id'].split('_')[1]) for x in json.load(open(R + '/data/agentgym/sciworld_test.json'))]
pool = sorted(set(range(len(g))) - set(test))
rng = random.Random(20261102)
tt = {g[i]['taskName'] for i in test}
p = [i for i in pool if g[i]['taskName'] in tt]; rng.shuffle(p)
dev = sorted(p[:60]); train = sorted(set(pool) - set(dev))
assert not set(dev) & set(test) and not set(train) & set(test)
name = lambda ids: Counter(g[i]['taskName'] for i in ids)
print('games', len(g), 'test', len(set(test)), 'dev', len(dev), 'train', len(train),
      'task types test', len(name(test)), 'dev', len(name(dev)))
json.dump({'test': test, 'dev': dev, 'train': train, 'seed': 20261102,
           'task': {str(i): g[i]['taskName'] for i in dev + test}}, open(R + '/data/splits/sciworld.json', 'w'))
