import random, statistics, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..')) + '/code/eval')
import evolve_x as X, repair
B = X.load_base(); F = [r for r in B.values() if not r['success']]; S = [r for r in B.values() if r['success']]
sz = [len(X.candidates(r)) for r in F]; old = [len(repair.candidates(r)) for r in F]
print('failures', len(F), 'C4 candidates mean %.1f median %d | C3 mean %.1f' % (statistics.mean(sz), statistics.median(sz), statistics.mean(old)))
rng = random.Random(0); pick = rng.sample(F, 6) + rng.sample(S, 4)
with ThreadPoolExecutor(10) as ex:
    res = list(ex.map(lambda r: repair.run(r['id']), pick))
same = sum([a for a, _ in x['trace']] == [a for a, _ in r['trace']] for x, r in zip(res, pick))
print('reproduce no-edit: success match', sum(x['success'] == r['success'] for x, r in zip(res, pick)), '/10; identical traces', same, '/10')
r = F[0]; C = X.candidates(r); e = X.as_edit([c for c in C if len(c) == 2][0])
x = repair.run(r['id'], e); print('edit', e); print('base plan', r['plan']); print('edit plan', x['plan'])
