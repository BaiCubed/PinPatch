"""Independent check: record frozen v1 TEST action sequences, replay them in a NEW TextCraft
instance (fresh crafting tree), and confirm success = goal item actually crafted within 50 steps."""
import sys, json, pickle, random, re
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinpatch_tc.paths import setup_sys_path, TEXTCRAFT_DIR
setup_sys_path()
import pinpatch_tc.envwrap as W
import pinpatch_tc.run as R
logs = {}
orig_step = W.Episode.step
def rec(self, a):
    out = orig_step(self, a); logs.setdefault(self.inst['id'], []).append(a); return out
W.Episode.step = rec
model = pickle.load(open(R.D + 'model.pkl', 'rb')); rng = random.Random(0)
res = {x['id']: R.run_episode(x, model, rng)['won'] for x in R.load('test')}
from textcraft.env import TextCraft
fresh = TextCraft(minecraft_dir=TEXTCRAFT_DIR)
ok = 0; mismatch = []
for x in R.load('test'):
    goal = re.fullmatch(r'craft (.+)\.', x['problem']).group(1)
    fresh.inventory = {}; fresh.goal = 'minecraft:' + goal.replace(' ', '_')
    won = False
    for i, a in enumerate(logs[x['id']][:50]):
        obs, r, term, _, _ = fresh.step(a)
        if r > 0 and obs.startswith('Crafted') and obs.endswith(fresh.goal):
            won = True; break
    ok += won
    if won != res[x['id']]: mismatch.append(x['id'])
print({'reported_success': sum(res.values()), 'replayed_success': ok, 'mismatches': mismatch,
       'max_actions': max(len(v) for v in logs.values())})
