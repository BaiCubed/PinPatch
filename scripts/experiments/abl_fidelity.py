"""Fidelity: current code (no ABL) must reproduce held-out episodes of the full system (ho2_gater2) action-for-action."""
import json, os, random, sys
from concurrent.futures import ThreadPoolExecutor
R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
os.environ.update(LOO_DEMO='1', META_SHIELD='both', LCH_TAU='0.9761', PATCHES=f'{R}/code/eval_frozen_c3r2/lib_gate_r2.json')
sys.path.insert(0, f'{R}/code/eval'); import argparse, run_eval as E
ref = {}
for s in range(8):
    for l in open(f'{R}/runs/ho2_gater2_s{s}/episodes.jsonl'):
        e = json.loads(l); ref[e['id']] = e
ids = random.Random(1).sample(sorted(ref), 24); P = [8123, 8124, 8125, 8126]
def go(j):
    i = ids[j]; a = argparse.Namespace(env='sciworld', llm_url=f'http://127.0.0.1:{P[j % 4]}/v1', seed=0, max_steps=30)
    return E.meta_replay(i, a)
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(go, range(24)))
same = sum(r['actions'] == ref[r['id']]['actions'] for r in res)
print('FIDELITY', same, '/ 24', 'success match', sum(r['success'] == ref[r['id']]['success'] for r in res))
sys.exit(0 if same == 24 else 1)
