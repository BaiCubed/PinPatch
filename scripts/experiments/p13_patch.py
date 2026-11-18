"""P1-3: make code/eval_frozen_c3r2_hot = frozen c3r2 + HOLD_TYPE (leave-one-type-out demo pool). Only retrieve() changes."""
import collections, json, shutil, os
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')); src, dst = f'{R}/code/eval_frozen_c3r2', f'{R}/code/eval_frozen_c3r2_hot'
if not os.path.exists(dst):
    shutil.copytree(src, dst)
p = f'{dst}/run_eval.py'; s = open(f'{src}/run_eval.py').read()
rep = [("def retrieve(task, k=2, exclude=None):", "def retrieve(task, k=2, exclude=None, drop_type=None):"),
       ("    demos = [d for d in json.load(open(f'{R}/data/splits/sciworld_demos.json')) if str(d['id']) != str(exclude)]\n",
        "    demos = [d for d in json.load(open(f'{R}/data/splits/sciworld_demos.json')) if str(d['id']) != str(exclude)]\n"
        "    if drop_type is not None:  # P1-3 (analysis): leave-one-type-out, no TRAIN demo of the episode's own task type\n"
        "        _gm = json.load(open(f'{R}/data/splits/sciworld_games.json'))\n"
        "        _fam = (lambda t: (t or '').split('-')[0] if os.environ.get('HOLD_TYPE') == 'family' else t)  # family = name before the first '-'\n"
        "        demos = [x for x in demos if _fam(_gm.get(str(x['id']))) != _fam(drop_type)]\n"),
       ("    cands = retrieve(task, 1 + rk + 8, exclude=idx if os.environ.get('LOO_DEMO') else None)",
        "    cands = retrieve(task, 1 + rk + 8, exclude=idx if os.environ.get('LOO_DEMO') else None,\n"
        "                     drop_type=json.load(open(f'{R}/data/splits/sciworld_games.json')).get(str(idx)) if os.environ.get('HOLD_TYPE') else None)")]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(p, 'w').write(s)
g = json.load(open(f'{R}/data/splits/sciworld_games.json')); d = json.load(open(f'{R}/data/splits/sciworld_demos.json'))
t = json.load(open(f'{R}/data/splits/sciworld.json'))['test']
tt = collections.Counter(g[str(i)] for i in t); dt = collections.Counter(g.get(str(x['id'])) for x in d)
fam = lambda x: (x or '').split('-')[0]; df = collections.Counter(fam(g.get(str(x['id']))) for x in d)
print('family demos kept per test type:', {k: len(d) - df[fam(k)] for k in tt})
print(len(d), 'demos;', len(tt), 'test types; demos per test type:', {k: dt.get(k, 0) for k in tt})
