"""Run methods from the MetaAct repository (https://github.com/BaiCubed/MetaAct: MetaAct-synth and its baselines) on the
same AgentGym environments, splits, step caps and serial Qwen2.5-7B server as every other method here.

Only the environment layer is replaced: each adapter wraps our harness env (run_eval.ENVS) and implements MetaAct's
TextEnv protocol (reset(task) -> (obs, info{goal}), step -> (obs, reward, done, info{won, score}),
admissible_actions(), score, max_steps). Port adaptations, all recorded in RESULTS.md:
  - ScienceWorld runs with AgentGym's settings (no simplification, so no teleport); the MetaAct prompts that
    describe 'teleport to <room>' are rewritten to AgentGym's 'go to <room>' mechanics.
  - Valid ScienceWorld actions come from a read-only /valid_actions route added to the AgentGym server.
  - Success/score follow AgentGym: final reward (ScienceWorld score >= 100, ALFWorld reward 1, TextCraft reward > 0,
    WebShop reward 1; WebShop score = 100 * reward).
  - The MetaAct LLM client turns a configured temperature of 0.0 into 0.2 (`float(x or 0.2)`); it is reset to 0.0.
  - An input longer than the server's max length ends the episode as a failure (as in agentenv), instead of the
    repo's default of dropping the episode from the results.
  - ScienceWorld 'teleport to <room>' (used by MetaAct's own prompts and RPMS's rules) is executed as a walk along
    the room graph learned from TRAIN demos (open door, go to, per hop); every primitive step counts toward the cap.
  - The step cap is enforced here: after max_steps env actions the episode is done (ReflAct's own loop did not
    check env.max_steps and ran up to 89 steps).
usage: metaact_ag.py ENV SPLIT RUN_ID PORT -- <args for the MetaAct script, e.g. --baseline react>
       (script chosen with MA_SCRIPT=run_episode_baselines|run_episode_eval|run_reflexion_eval, default baselines)"""
import json, os, re, sys

import run_eval as E

MA = E.R + '/code/MetaAct'
sys.path.insert(0, MA + '/src'); sys.path.insert(0, MA + '/scripts')
CAPS = dict(textcraft=40, alfworld=30, sciworld=30, webshop=10)
DS = dict(textcraft='textcraft', alfworld='alfworld', sciworld='sciworld', webshop='webshop')


def room_of(obs, last):
    m = (re.search(r'This (?:room|outside location) is called the ([a-z ]+?)\.', obs)
         or re.search(r'You move to the ([a-z ]+?)\.', obs))
    return m.group(1) if m else last


class AGEnv:
    def __init__(self, name):
        self.name, self.max_steps = name, CAPS[name]
        self.env = E.ENVS[name]()
        self._score, self._won, self._adm, self.timeouts, self.n_steps = 0.0, False, [], 0, 0

    # ---- reset ----
    def reset(self, task):
        i = task['ag_idx']; self._score, self._won, self.n_steps = 0.0, False, 0
        if self.name == 'sciworld':
            j = self.env.post('/reset', data_idx=i)
            goal = j['task_description'].strip()
            obs = f"{j['observation'].strip()}\n\nYour task is to: {goal}"
            self._room = room_of(j['observation'], None)
        elif self.name == 'alfworld':
            obs = self.env.reset(i).split('\nAVAILABLE ACTIONS: ')[0]
            m = re.search(r'Your task is to:\s*(.+)', obs); goal = m.group(1).strip() if m else ''
        elif self.name == 'webshop':
            obs = self.env.reset(i); goal = E.ws_instruction(obs)
        else:
            obs = self.env.reset(i).strip()
            m = re.search(r'Goal:\s*craft\s+(.+?)\.?\s*$', obs); goal = f'craft {m.group(1)}' if m else ''
            self._tc_space(obs)
        return obs, dict(won=False, goal=goal)

    # ---- step ----
    def step(self, action):
        act = (action or '').strip()
        if self.n_steps >= self.max_steps:  # the cap holds for every method, whatever its own loop does
            return '[step limit reached]', 0.0, True, dict(won=self._won, score=self._score)
        mt = re.match(r'teleport to (?:the )?(.+?)\.?$', act, re.I) if self.name == 'sciworld' else None
        if mt:  # AgentGym ScienceWorld has no teleport: walk the learned room graph, every primitive step counts
            return self._teleport(mt.group(1).strip().lower())
        obs, reward, done = self.env.step(act); self.n_steps += 1
        if self.name == 'sciworld':
            self._room = room_of(obs, getattr(self, '_room', None))
        info = {}
        if self.name == 'sciworld':
            self._score = float(reward); self._won = self._score >= 100
        elif self.name == 'webshop':
            self._score = 100.0 * float(reward); self._won = float(reward) >= 1
        else:
            self._won = bool(self.env.success(reward)); self._score = 100.0 * self._won
            if self.name == 'textcraft':
                info['invalid'] = obs.strip().startswith('Could not')
        info.update(won=self._won, score=self._score)
        return obs, float(reward or 0.0), bool(done or self._won), info

    def _teleport(self, dst):
        path = E.route(self._room, dst) if getattr(self, '_room', None) and dst in E.room_graph() else None
        if not path:
            obs, reward, done, info = self._step_raw(f'go to {dst}')
            return obs, reward, done, info
        out = None
        for nxt in path:
            for prim in (f'open door to {nxt}', f'go to {nxt}'):
                if self.n_steps >= self.max_steps:
                    return out or ('[step limit reached]', 0.0, True, dict(won=self._won, score=self._score))
                out = self._step_raw(prim)
                if out[2]:
                    return out
            if self._room != nxt:
                return out
        return out

    def _step_raw(self, act):
        obs, reward, done = self.env.step(act); self.n_steps += 1
        self._room = room_of(obs, getattr(self, '_room', None))
        self._score = float(reward); self._won = self._score >= 100
        return obs, float(reward or 0.0), bool(done or self._won), dict(won=self._won, score=self._score)

    # ---- admissible actions ----
    def admissible_actions(self):
        if self.name == 'alfworld':
            cmds = list(getattr(self.env, 'available', []))
            kept = [c for c in cmds if c not in ('look', 'inventory', 'help')]
            return kept + [c for c in ('look', 'inventory') if c in cmds]
        if self.name == 'sciworld':
            r = E.requests.get(self.env.url + '/valid_actions', params=dict(id=self.env.id), timeout=300)
            return r.json()[:2000] if r.ok else []
        if self.name == 'webshop':
            try:
                clk = self.env.available()
            except Exception:
                return []
            obs = self.env.observe()
            return (['search[<your query>]'] if obs.rstrip().endswith('[SEP] Search') or obs.strip().endswith('Search')
                    else []) + [f'click[{c}]' for c in clk]
        return self._tc_adm

    def _tc_space(self, obs):
        craft = [l.strip() for l in obs.split('\n') if l.strip().startswith('craft ') and ' using ' in l]
        tree, base, seen = E.TextCraft.tree, [], set()
        for line in craft:
            for part in line.split(' using ', 1)[1].split(','):
                name = re.sub(r'^\s*\d+\s+', '', part).strip()
                if not name or name in seen:
                    continue
                seen.add(name); iid = 'minecraft:' + name.replace(' ', '_')
                try:
                    if tree.is_tag(iid) or tree.is_craftable(iid) or not tree.is_valid_item(iid):
                        continue
                except Exception:
                    continue
                base.append(name)
        self._tc_adm = craft + [f'get 1 {b}' for b in sorted(base)] + ['inventory']

    @property
    def score(self):
        return self._score

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


SW_AG = ("Interact with a science-experiment environment (ScienceWorld) to solve a task. Only output one line per "
         "turn: either a single action, or 'think: ...' to reason. Mechanics: move between adjacent rooms with "
         "'go to <room>' (open a closed door first with 'open door to <room>'); 'look around' lists the room; "
         "'focus on <obj>' ONLY when the task explicitly asks for that exact object (a wrong focus ends the episode); "
         "'wait'/'wait1' passes time; devices need 'activate <device>'; measure with 'use thermometer on <obj>'; "
         "common rooms: kitchen, workshop, greenhouse, outside, bedroom, living room, bathroom, foundry, art studio, "
         "hallway.\n")


def train_slice(env, sp, n):
    """n TRAIN ids, seed 20261105; ScienceWorld round-robins over task types (one per type first), as sciworld_train30."""
    import random
    ids = sorted(sp['train']); random.Random(20261105).shuffle(ids)
    if env != 'sciworld':
        return sorted(ids[:n])
    path = f'{E.R}/data/splits/sciworld_games.json'  # data_idx -> task name, same enumeration as the AgentGym server
    types = {int(k): v for k, v in json.load(open(path)).items()}
    by = {}
    for i in ids:
        by.setdefault(types[i], []).append(i)
    out, k = [], 0
    while len(out) < n and any(len(v) > k for v in by.values()):
        out += [v[k] for t, v in sorted(by.items()) if len(v) > k][: n - len(out)]; k += 1
    return sorted(out)


def tasks_file(env, split):
    sp = json.load(open(f'{E.R}/data/splits/{env}.json'))
    typ = sp.get('task', {})
    if env == 'sciworld':  # train ids have no type in the split file; the game index has all of them
        typ = {**json.load(open(f'{E.R}/data/splits/sciworld_games.json')), **typ}
    if env == 'alfworld':
        mp = json.load(open(f'{E.R}/code/AgentGym/agentenv-alfworld/configs/mappings_train.json'))
        typ = {**{str(x['item_id']): x['task_type'].split('-')[0] for x in mp}, **typ}
    m = re.fullmatch(r'train(\d+)', split)
    if m:  # a fixed TRAIN slice for methods that learn before evaluation (same sizes as the MetaAct repo's slices)
        sp[split] = train_slice(env, sp, int(m.group(1)))
    out = f'{E.R}/runs/ma_tasks/{env}_{split}.jsonl'; os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        for i in sp[split]:
            f.write(json.dumps(dict(id=f'{env}_{i}', ag_idx=i, dataset=DS[env], task_type=typ.get(str(i), env))) + '\n')
    return out


def config_file(env, port):
    import yaml
    cfg = yaml.safe_load(open(f'{MA}/configs/qwen38b.yaml'))
    cfg['llm'].update(provider='openai', model=os.environ.get('LLM_MODEL', E.MODEL), base_url=f'http://127.0.0.1:{port}/v1', api_key_env='OPENAI_API_KEY',
                      supports_thinking=False, temperature=0.0, max_tokens=256, timeout_s=600, max_retries=4)
    cfg['env'].update(name=env, max_steps=CAPS[env])
    for k in ('sciworld', 'webshop', 'textcraft'):
        cfg.setdefault(k, {})['max_steps'] = CAPS[env]
    cfg['embeddings'] = dict(provider='local', model='sentence-transformers/all-MiniLM-L6-v2')
    out = f'{E.R}/runs/ma_tasks/cfg_{env}_{port}.yaml'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    yaml.safe_dump(cfg, open(out, 'w'))
    return out


def patch():
    import metaact.envs as me
    import metaact.baselines.embodied as emb
    import metaact.utils.llm as ml
    me.make_env = lambda task, cfg: AGEnv(next(k for k, v in DS.items() if v == task['dataset']))
    emb._SW_INSTRUCTION = SW_AG
    orig_init = ml._OpenAIChatClient.__init__

    def init(self, cfg):
        orig_init(self, cfg); self.default_temperature = float(cfg.temperature)
    ml._OpenAIChatClient.__init__ = init
    orig_chat = ml._OpenAIChatClient.chat

    def chat(self, messages, *a, **k):
        try:
            return orig_chat(self, messages, *a, **k)
        except RuntimeError as e:
            if 'maximum context length' in str(e) or 'max_model_len' in str(e) or "400" in str(e):
                raise ContextFull(str(e)) from e
            raise
    ml._OpenAIChatClient.chat = chat
    return me


class ContextFull(Exception):
    pass


def raw():
    """metaact_ag.py raw SCRIPT ENV PORT -- args: run one of the repo's own scripts (artifact building) with the same
    patches and an LLM config pointing at PORT."""
    script, env, port = sys.argv[2:5]
    rest = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')
    me = patch(); mod = __import__(script)
    if hasattr(mod, 'make_env'):
        mod.make_env = me.make_env
    if hasattr(mod, 'ART') and os.environ.get('MA_ART'):  # keep the repo's shipped artifacts untouched
        mod.ART = os.environ['MA_ART']
    sys.argv = [script + '.py', '--config', config_file(env, port)] + rest
    os.chdir(MA)
    mod.main()


def main():
    if sys.argv[1] == 'raw':
        return raw()
    env, split, run_id, port = sys.argv[1:5]
    rest = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')
    os.environ['HF_HUB_OFFLINE'] = os.environ.get('HF_HUB_OFFLINE', '0')
    me = patch()
    script = os.environ.get('MA_SCRIPT', 'run_episode_baselines')
    mod = __import__(script)
    mod.make_env = me.make_env
    # an episode whose prompt outgrows the server ends as a failure instead of being dropped
    if hasattr(mod, 'BASELINES'):
        for k, fn in list(mod.BASELINES.items()):
            def wrap(task, cfg, llm, envobj, _fn=fn, **kw):
                try:
                    return _fn(task, cfg, llm, envobj, **kw)
                except ContextFull:
                    return dict(success=bool(getattr(envobj, '_won', False)), steps=envobj.n_steps, turns=-1, invalid_actions=0,
                                latency_s=0, goal='', trajectory=[('[context full]', '')], context_full=True)
            mod.BASELINES[k] = wrap
    if script == 'run_episode_eval':
        import metaact.graph_ops.episode as ge
        orig = ge.EpisodeRunner.run_episode

        def run_episode(self, task, _orig=orig):
            try:
                return _orig(self, task)
            except ContextFull:
                e = self.env
                return dict(success=bool(getattr(e, '_won', False)), steps=e.n_steps, invalid_actions=0, latency_s=0,
                            retries_used=0, grounding_fixes=0, trajectory=[('[context full]', '')], goal='',
                            memory_final='', plan_final='', context_full=True)
        ge.EpisodeRunner.run_episode = run_episode
        mod.EpisodeRunner = ge.EpisodeRunner
    sys.argv = [script + '.py', '--config', config_file(env, port), '--dataset', tasks_file(env, split),
                '--run_id', run_id] + (['--max_turns', str(2 * CAPS[env] + 10)] if script == 'run_episode_baselines' else []) + rest
    os.chdir(MA)
    mod.main()


if __name__ == '__main__':
    main()
