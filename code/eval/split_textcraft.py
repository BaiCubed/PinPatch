"""TextCraft dev 60 / train rest from non-test AgentGym ids whose goals are not test goals (seed 20261102).
Also reports goal overlap with the legacy (ADaPT-line) train/dev splits. Run with the tc env python."""
import json, os, random, re, sys
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
AG = R + '/code/AgentGym/agentenv-textcraft'
sys.path.insert(0, AG); os.chdir(AG)
from agentenv_textcraft.crafting_tree import CraftingTree
from agentenv_textcraft.environment import TextCraftEnv
tree = CraftingTree(minecraft_dir='agentenv_textcraft/')
env = TextCraftEnv(crafting_tree=tree, commands=None, goal=None)
N = len(list(tree.item_recipes_min_depth(1)))
test = [int(x['item_id'].split('_')[1]) for x in json.load(open(R + '/data/agentgym/textcraft_test.json'))]
g = {i: re.search(r'Goal: craft (.+?)\.', env.reset(data_idx=i)[0]).group(1) for i in range(N)}
tg = {g[i] for i in test}
pool = [i for i in range(N) if i not in set(test) and g[i] not in tg]
p = list(pool); random.Random(20261102).shuffle(p)
dev, train = sorted(p[:60]), sorted(p[60:])
json.dump({'test': test, 'dev': dev, 'train': train, 'seed': 20261102}, open(R + '/data/splits/textcraft.json', 'w'))
L = R + '/code/pinpatch-textcraft/data/'
norm = lambda s: re.sub(r'^craft ', '', s.strip()).rstrip('.')
leg = {s: {norm(x['problem']) for x in json.load(open(L + f'{s}.json'))} for s in ('train', 'dev')}
print('goals', N, 'test', len(test), 'dev', len(dev), 'train', len(train))
for s in leg:
    print('AG test x legacy', s, len(tg & leg[s]))
