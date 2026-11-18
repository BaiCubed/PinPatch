"""TEST = the 200 ADaPT published instances. TRAIN/DEV = env.reset(seed) instances whose
goal is NOT any test goal; goals are split disjointly between TRAIN and DEV."""
import json, random, sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinpatch_tc.paths import ADAPT_RESULTS, DATA
from pinpatch_tc.envwrap import raw_env
d = json.load(open(ADAPT_RESULTS))
test = [{'id': f'env_{i}', 'problem': d[f'env_{i}']['problem'].strip(), 'commands': d[f'env_{i}']['commands']} for i in range(200)]
test_goals = {t['problem'] for t in test}
import time
from textcraft.utils import item_id_to_str
env = raw_env(); pool = {}
order = sorted(env.crafting_tree.item_recipes_min_depth(2), key=lambda x: -x[1])  # same rule as env.reset
for seed in range(1000, 1000 + len(order)):  # seed % len covers every goal exactly once
    name = 'craft ' + item_id_to_str(order[seed % len(order)][0]) + '.'
    if name in test_goals or name in pool: continue  # skip before the expensive reset
    t = time.time(); obs, _ = env.reset(seed=seed); c, p = obs.split('Goal: ')
    p = p.strip(); assert p == name
    pool[p] = {'id': f'seed_{seed}', 'problem': p, 'commands': c}
    print(len(pool), p, round(time.time() - t, 2), flush=True)
goals = sorted(pool); random.Random(0).shuffle(goals)
half = len(goals) // 2
train = [pool[g] for g in goals[:half]]; dev = [pool[g] for g in goals[half:]]
import os as _os
keep = _os.path.exists(DATA + 'train.json') and _os.path.exists(DATA + 'dev.json')
for name, x in ([('test', test)] if keep else [('test', test), ('train', train), ('dev', dev)]):
    json.dump(x, open(DATA + f'{name}.json', 'w'), indent=1)
print('unique test goals', len(test_goals), 'train', len(train), 'dev', len(dev),
      'overlap', len({t['problem'] for t in train} & test_goals))
