"""Train-only ensembling: pick the effect model with the best final TRAIN curve and give it the
union of recipes verified across all seeds. usage: PYTHONHASHSEED=0 merge_models.py WORKDIR"""
import json, os, pickle, sys
sys.path.insert(0, __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/code/pinpatch-textcraft')
from pinpatch_tc.agent import LiftedModel  # noqa: F401  (needed to unpickle)

wd = sys.argv[1].rstrip('/') + '/'
tags = [''] + [f'_s{s}' for s in range(1, 5)]
scored = []
for t in tags:
    c = json.load(open(wd + f'train_log{t}.json'))['curve'][-1]
    scored.append((-c['success'], c.get('fallback_actions', 0), t))
scored.sort()
best = scored[0][2]
print('train ranking', scored)
m = pickle.load(open(wd + f'model{best}.pkl', 'rb'))
union = {}
for t in tags:
    for a, n in getattr(pickle.load(open(wd + f'model{t}.pkl', 'rb')), 'verified', {}).items():
        union[a] = union.get(a, 0) + n
print('best', repr(best), 'verified own', len(m.verified), 'union', len(union))
m.verified = union
os.makedirs(wd + 'merged/results', exist_ok=True)
pickle.dump(m, open(wd + 'merged/model.pkl', 'wb'))
