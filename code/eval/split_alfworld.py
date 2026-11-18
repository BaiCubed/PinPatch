"""ALFWorld: test = official 200 (ids 2420-2619, valid_train); dev = 60 random TRAIN ids (seed 20261102); train = rest."""
import json, random
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
test = [int(x['item_id'].split('_')[1]) for x in json.load(open(R + '/data/agentgym/alfworld_test.json'))]
tr = json.load(open(R + '/code/AgentGym/agentenv-alfworld/configs/mappings_train.json'))
pool = [m['item_id'] for m in tr]
p = list(pool); random.Random(20261102).shuffle(p)
dev, train = sorted(p[:60]), sorted(p[60:])
assert not set(dev) & set(test) and not set(train) & set(test)
types = {m['item_id']: m['task_type'].split('-')[0] for m in tr}
te = {m['item_id']: m['task_type'].split('-')[0] for m in json.load(open(R + '/code/AgentGym/agentenv-alfworld/configs/mappings_test.json'))}
json.dump({'test': test, 'dev': dev, 'train': train, 'seed': 20261102, 'task': {str(i): (types.get(i) or te.get(i)) for i in dev + test}},
          open(R + '/data/splits/alfworld.json', 'w'))
print('test', len(test), min(test), max(test), 'dev', len(dev), 'train', len(train))
