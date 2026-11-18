"""WebShop split: test = AgentGym's 200 test sessions; dev = 60 and train = the rest of the 6910 synthetic goal
sessions (num_products=1000), drawn with a fixed seed and disjoint from test."""
import json, random
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
test = [int(x['item_id'].split('_')[1]) for x in json.load(open(R + '/data/agentgym/webshop_test.json'))]
rest = sorted(set(range(6910)) - set(test))
random.Random(20261104).shuffle(rest)
json.dump(dict(test=test, dev=sorted(rest[:60]), train=sorted(rest[60:])), open(R + '/data/splits/webshop.json', 'w'))
print(len(test), 60, len(rest) - 60)
