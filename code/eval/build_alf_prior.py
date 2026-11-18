"""Learn from ALFWorld TRAIN trajectories only (dev/test ids excluded):
  procedures[task_type]  = lifted high-level plan, args replaced by roles (obj / target / place:<class>)
  where[obj_class]       = counts of receptacle classes the object was picked up from (search prior)
Writes data/splits/alfworld_prior.json."""
import json, os
from collections import Counter, defaultdict
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
sp = json.load(open(R + '/data/splits/alfworld.json'))
train = set(sp['train'])
maps = json.load(open(R + '/code/AgentGym/agentenv-alfworld/configs/mappings_train.json'))
procs = defaultdict(Counter)
where = defaultdict(Counter)
for m in maps:
    if m['item_id'] not in train:
        continue
    f = os.path.join(R, 'data/alfworld/json_2.1.1/train', m['task_type'], m['task_id'], 'traj_data.json')
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    p = d['pddl_params']
    obj, tgt = p['object_target'].lower(), p['parent_target'].lower()
    role = lambda a: 'obj' if a == obj else 'target' if a == tgt else 'place:' + a
    steps, last_loc = [], None
    for h in d['plan']['high_pddl']:
        act, args = h['discrete_action']['action'], h['discrete_action']['args']
        if act == 'NoOp':
            continue
        if act == 'GotoLocation':
            last_loc = args[0]
        if act == 'PickupObject' and args and args[0] == obj and last_loc:
            where[obj][last_loc] += 1
        steps.append(act + '(' + ','.join(role(x) for x in args) + ')')
    # the location before a pickup is scene-specific; it becomes a prior-guided search
    lifted = []
    for i, s in enumerate(steps):
        nxt = steps[i + 1] if i + 1 < len(steps) else ''
        lifted.append('Search(obj)' if s.startswith('GotoLocation') and nxt.startswith('PickupObject(obj') else s)
    procs[d['task_type']][' ; '.join(lifted)] += 1
out = dict(procedures={t: c.most_common(1)[0][0] for t, c in procs.items()},
           procedure_support={t: [c.most_common(1)[0][1], sum(c.values())] for t, c in procs.items()},
           where={o: dict(c.most_common()) for o, c in where.items()})
json.dump(out, open(R + '/data/splits/alfworld_prior.json', 'w'), indent=1)
for t, pr in out['procedures'].items():
    print(t, out['procedure_support'][t], '|', pr)
print('objects', len(out['where']), 'mug ->', list(out['where'].get('mug', {}).items())[:5])
