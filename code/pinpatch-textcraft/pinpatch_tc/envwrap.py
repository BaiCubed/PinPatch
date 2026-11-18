"""TextCraft wrapper built from a published (problem, commands) pair.
Reset = empty inventory + goal parsed from the public problem text (same contract
as the project's published-input binding). Inventory queries are free (not counted
in the H50 action budget), matching the 031 protocol."""
import sys, re, json, random
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinpatch_tc.paths import setup_sys_path, TEXTCRAFT_DIR
setup_sys_path()
from textcraft.env import TextCraft

_ENV = None
def raw_env():
    global _ENV
    if _ENV is None:
        _ENV = TextCraft(minecraft_dir=TEXTCRAFT_DIR)
    return _ENV

class Episode:
    def __init__(self, inst, horizon=50):
        self.env = raw_env(); self.inst = inst; self.horizon = horizon
        self.env.inventory = {}
        goal = re.fullmatch(r'craft (.+)\.', inst['problem'].strip()).group(1)
        self.env.goal = 'minecraft:' + goal.replace(' ', '_')
        self.steps = 0; self.done = False; self.won = False; self.invalid = 0; self.log = []
    @property
    def observation(self):
        return self.inst['commands'] + 'Goal: ' + self.inst['problem']
    def inventory_text(self):
        return self.env.step('inventory')[0]
    def step(self, action):
        assert not self.done
        obs, r, term, _, _ = self.env.step(action)
        self.steps += 1
        ok = obs.startswith('Got ') or obs.startswith('Crafted ')
        self.invalid += (not ok)
        self.log.append((action, obs))
        if r > 0: self.won = True
        if term or self.steps >= self.horizon: self.done = True
        return obs, r, self.done, {'invalid': not ok, 'won': r > 0}

def parse_inventory(text):
    pairs = re.findall(r'\[([^\]]+)\]\s*\((\d+)\)', text)
    return {k: int(v) for k, v in pairs if int(v)}
