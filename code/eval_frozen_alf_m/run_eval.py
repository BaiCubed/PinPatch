"""Unified evaluation.
usage: run_eval.py --env textcraft --split dev --method react --seed 0 --max-steps 40 --run-id ID
Writes runs/<id>/episodes.jsonl (resumable) and runs/<id>/summary.json.
Deterministic only against the serial vLLM server (runs/setup/vllm_serial.sh).
"""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

R = __import__('os').environ.get('PINPATCH_ROOT', __import__('os').path.abspath(__file__ + '/../../..'))
MODEL = R + '/models/Qwen2.5-7B-Instruct'
LLM = dict(temperature=0.0, top_p=1.0, max_tokens=256)
reset_lock = threading.Lock()


class TextCraft:
    tree = None

    def __init__(self):
        ag = R + '/code/AgentGym/agentenv-textcraft'
        with reset_lock:
            if TextCraft.tree is None:  # the package loads recipes by relative path at import
                sys.path.insert(0, ag)
                cwd = os.getcwd(); os.chdir(ag)
                from agentenv_textcraft.crafting_tree import CraftingTree
                TextCraft.tree = CraftingTree(minecraft_dir='agentenv_textcraft/')
                os.chdir(cwd)
            from agentenv_textcraft.environment import TextCraftEnv
        self.env = TextCraftEnv(crafting_tree=TextCraft.tree, commands=None, goal=None)

    def reset(self, idx):
        with reset_lock:  # env.reset uses the global random module
            return self.env.reset(data_idx=idx)[0]

    def step(self, action):
        obs, r, term, trunc, _ = self.env.step(action)
        return obs, r, term or trunc

    def success(self, reward):
        return reward > 0

    def close(self):
        pass


class SciWorld:
    """AgentGym ScienceWorld server over HTTP. Success = score 100."""
    url = os.environ.get('SCIWORLD_URL', 'http://127.0.0.1:36102')

    def __init__(self):
        self.id = requests.post(self.url + '/create', timeout=300).json()['id']

    def post(self, path, **body):
        r = requests.post(self.url + path, json=dict(id=self.id, **body), timeout=300)
        r.raise_for_status()
        return r.json()

    def reset(self, idx):
        j = self.post('/reset', data_idx=idx)
        self.obs = j['task_description'] + '\n' + j['observation']
        return self.obs

    def step(self, action):
        j = self.post('/step', action=action)
        self.obs = j['observation']
        return j['observation'], j['score'], j['done']

    def success(self, reward):
        return reward >= 100

    def close(self):
        self.post('/close')


class TextCraftADaPT(TextCraft):
    """The 200 ADaPT published (problem, commands) instances, rebuilt in the AgentGym TextCraft env."""
    data = None

    def reset(self, idx):
        if TextCraftADaPT.data is None:
            TextCraftADaPT.data = json.load(open(f'{R}/data/splits/textcraft_adapt200.json'))
        d = TextCraftADaPT.data[str(idx)]
        goal = 'minecraft:' + re.fullmatch(r'craft (.+)\.', d['problem'].strip()).group(1).replace(' ', '_')
        cmds = re.sub(r'^Crafting commands:\s*', '', d['commands'].strip())  # the env adds this header itself
        with reset_lock:
            return self.env.reset(commands=cmds, goal=goal)[0]


class ALFWorld:
    """AgentGym ALFWorld server over HTTP, same observation format as the official client:
    the first observation lists AVAILABLE ACTIONS, later ones are the raw feedback. Success = reward 1."""
    url = os.environ.get('ALFWORLD_URL', 'http://127.0.0.1:36103')

    def __init__(self):
        self.id = requests.post(self.url + '/create', timeout=300).json()['id']

    def post(self, path, **body):
        r = requests.post(self.url + path, json=dict(id=self.id, **body), timeout=600)
        r.raise_for_status()
        return r.json()

    def reset(self, idx):
        j = self.post('/reset', game=idx, world_type='Text')
        self.available = j['available_actions']
        return j['observation'] + '\nAVAILABLE ACTIONS: ' + ','.join(j['available_actions'])

    def step(self, action):
        j = self.post('/step', action=action)
        self.available = j.get('available_actions', [])
        return j['observation'], j['reward'], j['done']

    def success(self, reward):
        return reward >= 1

    def close(self):
        pass  # the AgentGym ALFWorld server has no close route


ENVS = dict(textcraft=TextCraft, sciworld=SciWorld, textcraft_adapt=TextCraftADaPT, alfworld=ALFWorld)


def ids(env, split):
    return json.load(open(f'{R}/data/splits/{env}.json'))[split]


def intro(env):
    src = open(f'{R}/code/AgentGym/agentenv/agentenv/envs/{env.split("_")[0]}.py').read()  # textcraft_adapt -> textcraft prompt
    return eval(re.search(r'"value": (\'.*?\'|".*?"),\n', src, re.S).group(1))


def parse_textcraft(text):
    m = re.findall(r'Action:\s*(.*?)(?=\n|$)', text, re.DOTALL)
    if len(m) > 1:
        return None
    a = re.sub(r'[^A-Za-z0-9, ]+', '', m[-1] if m else '')
    return ' '.join(a.split()).strip()


def chat(url, messages, seed):
    body = dict(model=os.environ.get('LLM_MODEL', MODEL), messages=messages, seed=seed, **LLM)
    for t in range(6):
        try:
            r = requests.post(url + '/chat/completions', json=body, timeout=600)
            r.raise_for_status()
            j = r.json()
            return j['choices'][0]['message']['content'], j['usage']
        except requests.RequestException:
            if t == 5:
                raise
            time.sleep(2 ** t)


def react(idx, a):
    env = ENVS[a.env](); obs = env.reset(idx)
    msgs = [dict(role='user', content=intro(a.env)),
            dict(role='assistant', content="OK. I'll follow your instructions and try my best to solve the task."),
            dict(role='user', content=obs)]
    actions, reward, done, ptok, ctok, calls = [], 0, False, 0, 0, 0
    for step in range(a.max_steps):
        text, u = chat(a.llm_url, msgs, a.seed); calls += 1
        ptok += u['prompt_tokens']; ctok += u['completion_tokens']
        msgs.append(dict(role='assistant', content=text))
        act = parse_textcraft(text)
        if act is None:
            obs = "Error: Only one 'Action' is allowed per response. Please adjust your response."
        else:
            actions.append(act); obs, reward, done = env.step(act)
        msgs.append(dict(role='user', content=obs))
        if done:
            break
    env.close()
    return dict(id=idx, success=bool(env.success(reward)), score=reward, steps=step + 1, actions=actions,
                llm_calls=calls, prompt_tokens=ptok, completion_tokens=ctok)


def words(s):
    return set(re.findall(r'[a-z]+', s.lower()))


def retrieve(task, k=2):
    """Train demos whose task text is closest, most similar first; ties broken by id."""
    demos = json.load(open(f'{R}/data/splits/sciworld_demos.json'))
    w = words(task)
    return sorted(demos, key=lambda d: (-len(w & words(d['task'])) / len(w | words(d['task'])), d['id']))[:k]


def focus_targets(task):
    """Objects the task text names as focus targets; empty when the target must be discovered."""
    # a description such as "the animal with the longest life span" is a criterion, not an object name
    return [t.strip().lower() for t in re.findall(r'focus on (?:the )?([a-z0-9 ]+?)[.,]', task.lower())
            if t.strip() not in ('thing', 'it') and ' with ' not in t]


def ordered_focus(task):
    """Focus targets the task text requires in order ('First, focus on X. Next, focus on Y.')."""
    return [t.strip() for t in re.findall(r'(?:first|next|then),? focus on (?:the )?([a-z0-9 ]+?)[.,]', task.lower())
            if t.strip() not in ('thing', 'it') and ' with ' not in t]


def shielded(act, task, focused=None):
    m = re.match(r'focus on (.+)$', act.lower())
    if not m:
        return None
    obj = m.group(1)
    hit = lambda w: w in obj or obj.split(' in ')[0] in w
    names = focus_targets(task)
    if names and not any(hit(w) for w in names):
        return f'Blocked: the task only allows focusing on: {", ".join(names)}.'
    if focused is not None:
        order = ordered_focus(task)
        k = len(focused)
        if k < len(order) and not hit(order[k]):
            return f'Blocked: the task requires you to focus on the {order[k]} next.'
    return None


def react_demo(idx, a):
    env = ENVS[a.env](); obs = env.reset(idx); task = obs.split('\n')[0]
    demo = ''.join(f'\nExample task: {d["task"]}\nExample actions: ' + '; '.join(d['actions'][:30]) + '\n'
                   for d in retrieve(task))
    head = [dict(role='user', content=intro(a.env) + '\n\nSolved examples from similar tasks (objects and rooms may differ):' + demo),
            dict(role='assistant', content="OK. I'll follow your instructions and try my best to solve the task.")]
    hist = [dict(role='user', content=obs)]
    actions, reward, done, ptok, ctok, calls, blocked = [], 0, False, 0, 0, 0, 0
    for step in range(a.max_steps):
        text, u = chat(a.llm_url, head + [hist[0]] + hist[1:][-16:], a.seed); calls += 1
        ptok += u['prompt_tokens']; ctok += u['completion_tokens']
        hist.append(dict(role='assistant', content=text))
        m = re.findall(r'Action:\s*(.*?)(?=\n|$)', text, re.DOTALL)
        act = m[-1].strip() if m else ''
        actions.append(act)
        why = shielded(act, task)
        if why:
            obs = why; blocked += 1
        else:
            obs, reward, done = env.step(act)
        hist.append(dict(role='user', content=obs))
        if done:
            break
    env.close()
    return dict(id=idx, success=bool(env.success(reward)), score=reward, steps=step + 1, actions=actions,
                llm_calls=calls, prompt_tokens=ptok, completion_tokens=ctok, blocked=blocked)


_GRAPH = None


def room_graph():
    """Room adjacency learned from TRAIN expert paths: consecutive 'go to' targets are neighbours."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = {}
        for d in json.load(open(f'{R}/data/splits/sciworld_demos.json')):
            rooms = [a[6:] for a in d['actions'] if a.startswith('go to ')]
            for x, y in zip(rooms, rooms[1:]):
                _GRAPH.setdefault(x, set()).add(y); _GRAPH.setdefault(y, set()).add(x)
    return _GRAPH


def route(src, dst):
    """Shortest room path src -> dst on the learned graph (BFS, deterministic order)."""
    g, prev, q = room_graph(), {src: None}, [src]
    while q:
        x = q.pop(0)
        if x == dst:
            path = []
            while x != src:
                path.append(x); x = prev[x]
            return path[::-1]
        for y in sorted(g.get(x, ())):
            if y not in prev:
                prev[y] = x; q.append(y)
    return None


def compress(actions):
    """Collapse navigation runs in a demo into 'goto ROOM' macros."""
    out = []
    for a in actions:
        if a.startswith('open door to ') or a.startswith('go to '):
            room = a.split(' to ', 1)[1]
            if out and out[-1].startswith('goto '):
                out[-1] = 'goto ' + room if a.startswith('go to ') else out[-1]
            elif a.startswith('go to '):
                out.append('goto ' + room)
            continue
        out.append(a)
    return out


def current_room(obs, last):
    m = re.search(r'This room is called the ([a-z ]+?)\.', obs) or re.search(r'You move to the ([a-z ]+?)\.', obs)
    return m.group(1) if m else last


def meta_sci(idx, a):
    env = ENVS[a.env](); obs = env.reset(idx); task = obs.split('\n')[0]; room = current_room(obs, None)
    demo = ''.join(f'\nExample task: {d["task"]}\nExample actions: ' + '; '.join(compress(d['actions'])[:24]) + '\n'
                   for d in retrieve(task))
    rooms = ', '.join(sorted(room_graph()))
    head = [dict(role='user', content=intro(a.env)
                 + f'\nYou may also use the macro action "goto ROOM" to walk to any room ({rooms}); doors on the way are opened for you.'
                 + '\n\nSolved examples from similar tasks (objects and rooms may differ):' + demo),
            dict(role='assistant', content="OK. I'll follow your instructions and try my best to solve the task.")]
    hist = [dict(role='user', content=obs)]
    actions, reward, done, ptok, ctok, calls, blocked, macros = [], 0, False, 0, 0, 0, 0, 0
    steps, focused = 0, []
    while steps < a.max_steps and not done:
        text, u = chat(a.llm_url, head + [hist[0]] + hist[1:][-16:], a.seed); calls += 1
        ptok += u['prompt_tokens']; ctok += u['completion_tokens']
        hist.append(dict(role='assistant', content=text))
        m = re.findall(r'Action:\s*(.*?)(?=\n|$)', text, re.DOTALL)
        act = (m[-1] if m else '').split(';')[0].strip().rstrip('.')  # one action per turn
        act = act[:1].lower() + act[1:] if act.split(' ')[0].istitle() else act  # "Move x" -> "move x"; names keep case
        mg = re.match(r'goto (?:the )?([a-z ]+)$', act.lower())
        if mg and mg.group(1) in room_graph() and room:
            path = route(room, mg.group(1))
            if not path:
                steps += 1; actions.append(act)
                obs = (f'You are already in the {room}. Use "look around" to see what is here.' if path == []
                       else f'No known route from {room} to {mg.group(1)}.')
            else:
                macros += 1; obs = ''
                for nxt in path:
                    for prim in (f'open door to {nxt}', f'go to {nxt}'):
                        if steps >= a.max_steps or done:
                            break
                        o, reward, done = env.step(prim); steps += 1; actions.append(prim)
                    room = current_room(o, room) if 'You move to' in o or 'This room' in o else room
                    if room != nxt:
                        obs = o; break
                    obs = o
        else:
            why = shielded(act, task, focused)
            actions.append(act); steps += 1
            if why:
                obs = why; blocked += 1
            else:
                obs, reward, done = env.step(act)
                room = current_room(obs, room)
                if act.lower().startswith('focus on ') and 'You focus on' in obs:
                    focused.append(act[9:])
        hist.append(dict(role='user', content=obs))
    env.close()
    return dict(id=idx, success=bool(env.success(reward)), score=reward, steps=steps, actions=actions,
                llm_calls=calls, prompt_tokens=ptok, completion_tokens=ctok, blocked=blocked, macros=macros)


FAIL = ('No known action matches', 'Unknown action', 'not found', "can't", 'cannot', 'There is no',
        "isn't", 'is not', 'not possible', 'Blocked:', 'No known route')


def lift_map(src_task, dst_task):
    """Phrase substitutions src -> dst from a word-level diff of the two task texts (the lifted slots)."""
    import difflib
    a, b = src_task.split(), dst_task.split()
    subs = {}
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == 'replace':
            x, y = ' '.join(a[i1:i2]).strip('.,\''), ' '.join(b[j1:j2]).strip('.,\'')
            if x and y and x.lower() != y.lower():
                subs[x] = y
    # a differing box label ("green" vs "blue" box) is chosen at decision time, and substituting the bare
    # colour word would also rewrite unrelated objects such as wires
    return {x: y for x, y in subs.items() if not (f'{x} box' in src_task and f'{y} box' in dst_task)}


def apply_subs(action, subs):
    for x in sorted(subs, key=len, reverse=True):  # longest phrase first
        action = re.sub(r'\b' + re.escape(x) + r'\b', subs[x], action)
        action = re.sub(r'\b' + re.escape(x.lower()) + r'\b', subs[x].lower(), action)
    return action


def visible_objects(obs):
    """Object phrases listed in a 'look around' description ('a metal pot', 'the red box', 'baby wolf')."""
    out = []
    for part in re.split(r'\t|\n|, |\. |: ', obs):
        part = re.sub(r'^(?:an? |the |some )', '', part.strip().rstrip('.'))
        part = re.sub(r' \(containing .*$| \(.*\)$', '', part)
        if 1 <= len(part.split()) <= 5 and not part.startswith(('This ', 'In it', 'You ', 'Here ', 'A door', 'door')):
            out.append(part)
    return out


def ground(action, obs):
    """Replace the action's object phrase with the closest object visible in obs (word overlap); None if no gain."""
    objs = visible_objects(obs)
    m = re.match(r'(focus on|pick up|look at|examine|eat|open|close|activate|deactivate|mix|read) (.+?)( in inventory)?$', action)
    if not m or not objs:
        return None
    w = words(m.group(2))
    best = max(objs, key=lambda o: (len(w & words(o)) / len(w | words(o)), -len(o)))
    if not w & words(best) or best == m.group(2):
        return None
    return f'{m.group(1)} {best}'


def expand_loops(plan):
    """A pair (A, B) repeated 2+ times in a row in a demo is a measure-until-change loop: <loop A|B>."""
    out, i = [], 0
    while i < len(plan):
        a2 = plan[i:i + 2]
        n = 1
        while len(a2) == 2 and plan[i + 2 * n:i + 2 * n + 2] == a2:
            n += 1
        if n >= 2 and len(a2) == 2:
            first_is_probe = a2[0].startswith(('examine', 'look at'))
            act, probe = (a2[1], a2[0]) if first_is_probe else (a2[0], a2[1])
            out.append(f'<loop {act}|{probe}>'); i += 2 * n
        else:
            out.append(plan[i]); i += 1
    return out


def align_same_kind(plan, seen):
    """Consistent renaming of plan objects that are absent from the room onto present objects of the same
    kind (same head noun), in order of first appearance: 'blue wire'->'yellow wire', 'black wire'->'red wire'."""
    objs = visible_objects(seen)
    vis = set(objs)
    phrases = []
    for act in plan:
        for m in re.finditer(r'\b([a-z]+ (?:wire|light bulb|bulb|box|battery|motor|buzzer))\b', act):
            ph = m.group(1)
            if ph not in phrases and ph not in vis and not any(ph in o for o in objs):
                phrases.append(ph)
    used, mapping = set(p for p in phrases), {}
    # hand-given prior (declared in the report): circuit loads are interchangeable in a conductivity test
    loads = ('light bulb', 'motor', 'buzzer')
    kind = lambda x: 'load' if x.endswith(loads) else x.split(' ', 1)[1]
    for ph in phrases:
        cand = [o for o in objs if ' ' in o and kind(o) == kind(ph) and o not in used and not any(o == x for x in mapping.values())
                and not any(o in a2 for a2 in plan)]
        if cand:
            mapping[ph] = cand[0]
    return mapping


def boxes(task):
    return sorted(set(re.findall(r'(?:the )?([a-z]+ box)', task.lower())))


def meta_replay(idx, a):
    """Lifted procedure from the closest TRAIN demo, executed by a controller; Qwen is asked only when an
    action fails or a decision (which box) is due. Every primitive env step counts toward the cap."""
    env = ENVS[a.env](); obs = env.reset(idx); task = obs.split('\n')[0]; room = current_room(obs, None)
    d = retrieve(task, 1)[0]
    subs = lift_map(d['task'], task)
    plan = [apply_subs(x, subs) for x in compress(d['actions'])]
    hist, actions = [obs], []
    st = dict(reward=0, done=False, steps=0, calls=0, ptok=0, ctok=0, llm_steps=0, decisions=0, grounded=0)
    focused, trace = [], []

    def step(act):
        nonlocal room
        mg = re.match(r'go to (?:the )?([a-z ]+)$', act)
        if mg and room and mg.group(1) in room_graph() and mg.group(1) not in room_graph().get(room, ()) and mg.group(1) != room:
            goto(mg.group(1)); return hist[-1]
        why = shielded(act, task, focused)
        mf = re.match(r'focus on (.+?)(?: in inventory)?$', act)
        if mf and any(mf.group(1) in f for f in focused):
            why = f'Blocked: you have already focused on the {mf.group(1)}; do the next part of the task.'
        if why:
            o = why
        else:
            o, st['reward'], st['done'] = env.step(act)
            if o.startswith('Ambiguous request'):
                o, st['reward'], st['done'] = env.step('0')
            room = current_room(o, room)
            if act.startswith('focus on ') and 'You focus on' in o:
                focused.append(act[9:])
        st['steps'] += 1; actions.append(act); hist.append(f'> {act}\n{o}'); trace.append([act, o[:300]])
        if mg and 'door is not open' in o and st['steps'] + 1 < a.max_steps:
            step(f'open door to {mg.group(1)}'); return step(act)
        return o

    def goto(dst):
        path = route(room, dst) if room else None
        for nxt in path or []:
            for prim in (f'open door to {nxt}', f'go to {nxt}'):
                if st['steps'] >= a.max_steps or st['done']:
                    return
                step(prim)
            if room != nxt:
                return

    def ask(question):
        msgs = [dict(role='user', content=intro(a.env)), dict(role='assistant', content='OK.'),
                dict(role='user', content=task + '\n\nRecent history:\n' + '\n'.join(hist[-12:])[-6000:] + '\n\n' + question)]
        text, u = chat(a.llm_url, msgs, a.seed)
        st['calls'] += 1; st['ptok'] += u['prompt_tokens']; st['ctok'] += u['completion_tokens']
        m = re.findall(r'Action:\s*(.*?)(?=\n|$)', text, re.DOTALL)
        act = (m[-1] if m else text.strip().split('\n')[-1]).split(';')[0].strip().rstrip('.')
        return re.sub(r'^(focus on|pick up|move|go to) (?:\1 )?(?:the )?', r'\1 ', act)  # "focus on focus on the x" -> "focus on x"

    bx = boxes(task)
    plan = expand_loops(plan)
    for k, planned in enumerate(plan):
        if planned.startswith('<loop '):  # <loop ACTION|PROBE>: repeat until the probe's text changes
            act, probe = planned[6:-1].split('|')
            first, top = None, max([float(x) for x in re.findall(r'(-?\d+(?:\.\d+)?) degrees', task)] or [1e9])
            while st['steps'] + 2 <= a.max_steps and not st['done']:
                r = step(act); o = step(probe)
                first = first or o
                temps = [float(x) for x in re.findall(r'temperature of (-?\d+(?:\.\d+)?)', r)]
                if o != first or 'No known action' in o or (temps and temps[-1] > top):
                    break  # state changed, or the reading already settles every condition in the task
            continue
        if st['steps'] >= a.max_steps or st['done']:
            break
        if planned.startswith('goto '):
            goto(planned[5:]); continue
        used = [b for b in bx if b in planned]
        if used and len(bx) > 1:  # a decision: which box depends on what was observed
            opts = [planned.replace(used[0], b) for b in bx]
            choice = ask('Which action is correct now? Options:\n' + '\n'.join(opts)
                         + '\nFirst write "Thought:" quoting the measurement or observation that decides it and the matching '
                         'condition from the task, then "Action:" followed by exactly one option.')
            st['decisions'] += 1
            planned = next((o for o in opts if o in choice), planned)
        mf = re.match(r'focus on (.+?)(?: in inventory)?$', planned)
        if mf and any(mf.group(1) in f for f in focused):
            continue  # already focused on it; a repeated focus only wastes a step
        o = step(planned)
        if 'No known action matches' in o:  # ground the object name in what the room actually shows
            seen = next((h for h in reversed(hist) if 'Here you see' in h or 'In it, you see' in h), '')
            mp = align_same_kind(plan[k:], seen)
            if mp:  # rename consistently across the rest of the plan, then retry this action
                for x in sorted(mp, key=len, reverse=True):
                    plan[k:] = [re.sub(r'\b' + re.escape(x) + r'\b', mp[x], y) for y in plan[k:]]
                st['aligned'] = st.get('aligned', 0) + len(mp)
                if plan[k] != planned and st['steps'] < a.max_steps and not st['done']:
                    planned = plan[k]; o = step(planned)
            g = ground(planned, seen)
            mv = re.match(r'(focus on|pick up) (.+?)( in inventory)?$', planned)
            if not g and mv and visible_objects(seen):
                opts = [f'{mv.group(1)} {x}' for x in visible_objects(seen) if x not in ('agent', 'substance called air')]
                choice = ask('The remembered object is not here. Which visible object fits the task? Options:\n'
                             + '\n'.join(opts) + '\nFirst write "Thought:" with the reason, then "Action:" and exactly one option.')
                st['decisions'] += 1
                g = (next((x for x in opts if x == choice), None) or next((x for x in opts if x in choice), None)
                     or next((x for x in opts if x.split(' ', 2)[2] in choice), None))
                need_focus = 'focus on the thing' in task.lower() and not focused
                if g and mv.group(1) == 'pick up' and need_focus and st['steps'] < a.max_steps:
                    step('focus on ' + g.split(' ', 2)[2])
                if g and mv.group(2) in ' '.join(plan):  # keep later steps about the same object consistent
                    plan[:] = [x.replace(mv.group(2), g.split(' ', 2)[2]) for x in plan]
            if g and st['steps'] < a.max_steps and not st['done']:
                o = step(g); st['grounded'] += 1
        tries, failed = 0, {planned}
        while any(f in o for f in FAIL) and tries < 2 and st['steps'] < a.max_steps and not st['done']:
            act = ask(f'The planned action "{planned}" failed. Give the single next action that achieves the same purpose. '
                      'Answer in the format "Action: <action>".')
            st['llm_steps'] += 1; tries += 1
            if act in failed:
                break  # the repair repeats a failed action: move on with the plan
            failed.add(act)
            mg = re.match(r'goto (?:the )?([a-z ]+)$', act.lower())
            if mg and mg.group(1) in room_graph():
                goto(mg.group(1)); break
            o = step(act)
    while st['steps'] < a.max_steps and not st['done']:  # plan exhausted: let Qwen finish
        act = ask('The remembered procedure is finished but the task is not done. Give the single next action. '
                  'Answer in the format "Action: <action>".')
        st['llm_steps'] += 1
        mg = re.match(r'goto (?:the )?([a-z ]+)$', act.lower())
        before = st['steps']
        if mg and mg.group(1) in room_graph():
            goto(mg.group(1))
        if st['steps'] == before:  # not a usable goto (or already there): spend the step on the raw action
            step(act)
    env.close()
    return dict(id=idx, success=bool(env.success(st['reward'])), score=st['reward'], steps=st['steps'], actions=actions,
                llm_calls=st['calls'], prompt_tokens=st['ptok'], completion_tokens=st['ctok'], llm_steps=st['llm_steps'],
                decisions=st['decisions'], grounded=st['grounded'], demo=d['id'], subs=subs, trace=trace)


ALF_GOALS = [  # hand-written goal parser (declared prior): (regex, task_type)
    (r'(?:put|place) (?:a |an |some )?clean (\w+) (?:in|on|into) (?:the )?(\w+)', 'pick_clean_then_place_in_recep'),
    (r'clean (?:a |an |some |the )?(\w+) and put it (?:in|on|into) (?:the )?(\w+)', 'pick_clean_then_place_in_recep'),
    (r'(?:put|place) (?:a |an |some )?hot (\w+) (?:in|on|into) (?:the )?(\w+)', 'pick_heat_then_place_in_recep'),
    (r'heat (?:a |an |some |the )?(\w+) and put it (?:in|on|into) (?:the )?(\w+)', 'pick_heat_then_place_in_recep'),
    (r'(?:put|place) (?:a |an |some )?cool (\w+) (?:in|on|into) (?:the )?(\w+)', 'pick_cool_then_place_in_recep'),
    (r'cool (?:a |an |some |the )?(\w+) and put it (?:in|on|into) (?:the )?(\w+)', 'pick_cool_then_place_in_recep'),
    (r'(?:look at|examine) (?:a |an |some |the )?(\w+) (?:under|with) (?:the )?(desklamp)', 'look_at_obj_in_light'),
    (r'(?:put|place) two (\w+) (?:in|on|into) (?:the )?(\w+)', 'pick_two_obj_and_place'),
    (r'find two (\w+) and put them (?:in|on|into) (?:the )?(\w+)', 'pick_two_obj_and_place'),
    (r'(?:put|place) (?:a |an |some )?(\w+) (?:in|on|into) (?:the )?(\w+)', 'pick_and_place_simple'),
]
_ALF = None


def alf_prior():
    global _ALF
    if _ALF is None:
        _ALF = json.load(open(f'{R}/data/splits/alfworld_prior.json'))
    return _ALF


def meta_alf(idx, a):
    """PinPatch on ALFWorld: lifted TRAIN procedure for the task type, executed against the env's
    admissible actions; object search ordered by the TRAIN location prior; Qwen only when stuck."""
    env = ENVS[a.env](); obs = env.reset(idx)
    goal = re.search(r'Your task is to: (.+?)\.?(?:\n|$)', obs).group(1).lower()
    ttype, obj, tgt = None, None, None
    for rx, t in ALF_GOALS:
        m = re.search(rx, goal)
        if m:
            ttype, obj, tgt = t, m.group(1), m.group(2); break
    pri = alf_prior()
    plan = pri['procedures'].get(ttype, '').split(' ; ') if ttype else []
    st = dict(reward=0, done=False, steps=0, calls=0, ptok=0, ctok=0, llm_steps=0)
    actions, trace, hist = [], [], [obs]
    here, holding, placed = None, None, []

    def avail():
        return env.available

    def step(act):
        nonlocal here
        o, st['reward'], st['done'] = env.step(act)
        st['steps'] += 1; actions.append(act); trace.append([act, o[:300]]); hist.append(f'> {act}\n{o}')
        if act.startswith('go to ') and 'Nothing happens' not in o:
            here = act[6:]
        return o

    def live():
        return st['steps'] < a.max_steps and not st['done']

    def receps():
        return [x[6:] for x in avail() if x.startswith('go to ')]

    def open_here(o):
        if here and f'open {here}' in avail() and ('is closed' in o or 'closed' in o):
            o = step(f'open {here}')
        return o

    def seen_inst(o, cls):
        return [x for x in re.findall(r'\b(' + re.escape(cls) + r' \d+)', o) if x not in placed]

    def search(cls, want_use=False):
        order = pri['where'].get(cls, {})
        rs = receps()
        rs.sort(key=lambda r: (-order.get(re.sub(r' \d+$', '', r), 0), int(re.search(r'(\d+)$', r).group(1)) if re.search(r'\d+$', r) else 0, r))
        for r in rs:
            if not live():
                return None
            o = open_here(step(f'go to {r}'))
            if want_use and any(x.startswith('use ' + cls) for x in avail()):
                return next(x for x in avail() if x.startswith('use ' + cls))
            inst = seen_inst(o, cls)
            if inst:
                return inst[0]
        return None

    def goto_class(cls):
        if here and re.sub(r' \d+$', '', here) == cls:
            return True
        cand = sorted([r for r in receps() if re.sub(r' \d+$', '', r) == cls], key=lambda r: int(r.split()[-1]) if r.split()[-1].isdigit() else 0)
        if not cand:
            return False
        open_here(step(f'go to {cand[0]}'))
        return True

    def pick(act_prefix, pat):
        c = [x for x in avail() if x.startswith(act_prefix) and re.search(pat, x)]
        return c[0] if c else None

    def ask_llm(why):
        msgs = [dict(role='user', content=intro('alfworld')), dict(role='assistant', content='OK.'),
                dict(role='user', content=goal + '\n\nRecent history:\n' + '\n'.join(hist[-10:])[-5000:]
                     + f'\n\n{why}\nAVAILABLE ACTIONS: ' + ','.join(avail()) + '\nAnswer with "Action:" and one available action.')]
        text, u = chat(a.llm_url, msgs, a.seed)
        st['calls'] += 1; st['ptok'] += u['prompt_tokens']; st['ctok'] += u['completion_tokens']; st['llm_steps'] += 1
        m = re.findall(r'Action:\s*(.*?)(?=\n|$)', text, re.DOTALL)
        act = (m[-1] if m else '').strip().rstrip('.')
        return act if act in avail() else None

    def do(stage):
        nonlocal holding
        name, args = re.match(r'(\w+)\((.*)\)', stage).groups()
        if name == 'Search':
            inst = search(obj)
            if inst:
                holding = inst
            return bool(inst)
        if name == 'PickupObject':
            act = pick('take ', r'^take ' + re.escape(holding or obj) + r'\b') if holding else pick('take ', r'^take ' + re.escape(obj) + r' \d+')
            if act:
                o = step(act)
                holding = act.split(' from ')[0][5:]
                return 'Nothing happens' not in o
            return False
        if name == 'GotoLocation':
            place = args.split(':', 1)[1] if args.startswith('place:') else tgt
            if place == 'desklamp':
                return bool(search('desklamp', want_use=True))
            return goto_class(place)
        if name in ('CleanObject', 'HeatObject', 'CoolObject'):
            verb = name[:-6].lower()
            act = pick(verb + ' ', r'^' + verb + ' ' + re.escape(holding or obj))
            return bool(act) and 'Nothing happens' not in step(act)
        if name == 'ToggleObject':
            act = pick('use ', r'^use desklamp')
            return bool(act) and 'Nothing happens' not in step(act)
        if name == 'PutObject':
            act = pick('', r'^(?:move|put) ' + re.escape(holding or obj) + r' (?:to|in|on|in/on) ')
            if act:
                o = step(act)
                if 'Nothing happens' not in o:
                    placed.append(holding); holding = None
                    return True
            return False
        return False

    for stage in plan:
        if not live():
            break
        ok = do(stage)
        tries = 0
        while not ok and live() and tries < 3:  # stuck: one LLM-chosen admissible action, then retry the stage
            act = ask_llm(f'The step "{stage}" could not be done.')
            tries += 1
            if act:
                step(act)
            ok = do(stage) if live() else False
    while live():  # plan finished without success: plain ReAct-style fallback over admissible actions
        act = ask_llm('Continue the task.')
        if not act:
            break
        step(act)
    env.close()
    return dict(id=idx, success=bool(env.success(st['reward'])), score=st['reward'], steps=st['steps'], actions=actions,
                llm_calls=st['calls'], prompt_tokens=st['ptok'], completion_tokens=st['ctok'], llm_steps=st['llm_steps'],
                task_type=ttype, obj=obj, target=tgt, trace=trace)


METHODS = dict(react=react, react_demo=react_demo, meta_sci=meta_sci, meta_replay=meta_replay, meta_alf=meta_alf)


def main():
    p = argparse.ArgumentParser()
    for k, d in (('--env', 'textcraft'), ('--split', 'dev'), ('--method', 'react'), ('--run-id', ''),
                 ('--llm-url', 'http://127.0.0.1:8123/v1')):
        p.add_argument(k, default=d)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=40)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    out = f'{R}/runs/{a.run_id}'
    os.makedirs(out + '/logs', exist_ok=True)
    f = out + '/episodes.jsonl'; done = {}
    if os.path.exists(f):
        for l in open(f):
            try:
                r = json.loads(l); done[r['id']] = r
            except json.JSONDecodeError:  # a torn last line from an interrupted run is redone
                pass
    all_ids = ids(a.env, a.split)
    todo = [i for i in (all_ids[:a.limit] if a.limit else all_ids) if i not in done]
    lock = threading.Lock(); t0 = time.time()

    def work(i):
        r = METHODS[a.method](i, a)
        with lock:
            done[i] = r
            with open(f, 'a') as w:
                w.write(json.dumps(r) + '\n')
            print(len(done), i, r['success'], r['steps'], flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    rs = list(done.values()); n = sum(x['success'] for x in rs)
    s = dict(vars(a), n=len(rs), success=n, rate=n / max(1, len(rs)), llm=LLM, model=MODEL, seconds=time.time() - t0,
             mean_steps=sum(x['steps'] for x in rs) / max(1, len(rs)), llm_calls=sum(x['llm_calls'] for x in rs),
             mean_score=sum(x.get('score', 0) for x in rs) / max(1, len(rs)),
             prompt_tokens=sum(x['prompt_tokens'] for x in rs), completion_tokens=sum(x['completion_tokens'] for x in rs))
    json.dump(s, open(out + '/summary.json', 'w'), indent=1, sort_keys=True)
    print('SUMMARY', a.env, a.split, a.method, n, '/', len(rs), round(s['rate'], 3), flush=True)


if __name__ == '__main__':
    main()
