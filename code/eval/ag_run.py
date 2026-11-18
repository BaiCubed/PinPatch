"""PinPatch on the official AgentGym TextCraft env, H=40, no free inventory oracle:
the agent tracks inventory only from 'Got'/'Crafted' feedback, so every env call costs a step.
usage: PYTHONHASHSEED=0 ag_run.py WORKDIR train SEED | eval SPLIT ARM SEED | noop
env: HIDE_RECIPES=1 drops one intermediate recipe line per task (hidden-recipe dev); LLM_URL for llm arms.
"""
import json, os, re, sys, random
from collections import Counter

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
sys.path.insert(0, R + '/code/eval')
sys.path.insert(0, R + '/code/pinpatch-textcraft')
assert os.environ.get('PYTHONHASHSEED') == '0'
work = sys.argv[1].rstrip('/') + '/'
os.makedirs(work + 'results', exist_ok=True)

from run_eval import TextCraft
from pinpatch_tc import run
from pinpatch_tc.agent import parse_action, parse_commands

HIDE = os.environ.get('HIDE_RECIPES') == '1'
HIDDEN = {}


def hide_one(obs, idx):
    """Drop one intermediate item's recipe line from the visible commands (the env still accepts it)."""
    cmds, goal = parse_commands(obs)
    used = {e for _, (v, s) in cmds for r, n, e in s if r == 1}
    cand = sorted(l for l, (v, s) in cmds if s[0][2] != goal and s[0][2] in used)
    if not cand:
        return obs
    line = random.Random(f'hide{idx}').choice(cand)
    HIDDEN[idx] = line
    return obs.replace(line + '\n', '')


def llm_prior(entity, model):
    """Ask Qwen for candidate recipes; the env verifies them when the planner tries them."""
    from run_eval import chat
    q = (f'In Minecraft, how do you craft {entity}? Answer with up to 3 candidate recipes, one per line, '
         f'each exactly in the format: craft <count> {entity} using <count> <ingredient>, <count> <ingredient>')
    text, _ = chat(os.environ['LLM_URL'], [dict(role='user', content=q)], 0)
    out = []
    for l in text.splitlines():
        m = re.search(r'craft \d+ [a-z ]+ using \d+ [a-z ]+(, \d+ [a-z ]+)*', l.lower())
        p = parse_action(m.group(0)) if m else None
        if p and any(e == entity and r == 0 for r, n, e in p[1]):
            out.append((m.group(0), p[0], p[1]))
    return out


class AGEpisode:
    def __init__(self, inst, horizon=40):
        self.env = TextCraft(); self.observation = self.env.reset(inst['id'])
        if HIDE:
            self.observation = hide_one(self.observation, inst['id'])
        self.horizon = horizon; self.inv = Counter()
        self.steps = 0; self.done = False; self.won = False; self.invalid = 0; self.log = []

    def inventory_text(self):
        return 'Inventory: ' + ' '.join(f'[{k}] ({v})' for k, v in sorted(self.inv.items()) if v > 0)

    def _track(self, action, obs):
        m = re.match(r'Got (\d+) (.+)$', obs)
        if m:
            self.inv[m.group(2)] += int(m.group(1)); return True
        m = re.match(r'Crafted (\d+) minecraft:(.+)$', obs)
        if m:
            p = parse_action(action)
            for role, n, e in (p[1] if p else []):
                if role == 1:
                    self.inv[e] -= n
            self.inv[m.group(2).replace('_', ' ')] += int(m.group(1)); return True
        return False

    def step(self, action):
        obs, r, term = self.env.step(action)
        self.steps += 1
        ok = self._track(action, obs)
        self.invalid += (not ok)
        self.log.append((action, obs))
        if r > 0:
            self.won = True
        if term or self.steps >= self.horizon:
            self.done = True
        return obs, r, self.done, {'invalid': not ok, 'won': r > 0}


split = json.load(open(R + '/data/splits/textcraft.json'))
run.Episode = AGEpisode
run.H = 40
run.D = work
run.load = lambda s: [{'id': i} for i in split[s]][: int(os.environ.get('TC_LIMIT', 10 ** 9))]  # TC_LIMIT: first N ids only (release smoke test)
run.PRIOR = llm_prior

if sys.argv[2] == 'noop':
    pass
elif sys.argv[2] == 'train':
    sd = int(sys.argv[3])
    run.train(seed=sd, tag='' if sd == 0 else f'_s{sd}')
else:
    s, arm, sd = sys.argv[3], sys.argv[4], int(sys.argv[5])
    assert s != 'test' or os.path.exists(work + 'FREEZE.json'), 'test only after FREEZE.json'
    out = run.evaluate(s, arm, tag='' if sd == 0 else f'_s{sd}')
    print('FAILED', [(e['id'], e['goal'], e['steps']) for e in out['episodes'] if not e['won']], flush=True)
    print('hidden', len(HIDDEN), 'llm_calls', sum(e.get('llm_calls', 0) for e in out['episodes']), flush=True)
