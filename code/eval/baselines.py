"""Recent same-backbone baselines, reimplemented on the shared harness (not the authors' code):
  stateact  - StateAct (Rozanov & Rei, 2024): goal restated + explicit state tracking every turn
  reflexion - Reflexion (Shinn et al., 2023): verbal self-reflection after a failed trial, 2 trials
  adapt     - ADaPT (Prasad et al., 2024): executor; on self-reported failure the planner decomposes, depth <= 3
  expel     - ExpeL (Zhao et al., 2024): insights distilled from its own TRAIN trajectories + retrieved successes
All use the env's official action parser, temperature 0, and the same step cap as ReAct (per trial for Reflexion)."""
import json, os, random, re
import run_eval as E

OK = "OK. I'll follow your instructions and try my best to solve the task."


def parse(env, text):
    return E.parse_textcraft(text) if env.startswith('textcraft') else E.parse_react_official(text)


def goal_of(env, obs):
    m = (re.search(r'Goal: (.+?)(?:\n|$)', obs) or re.search(r'Your task is to: (.+?)(?:\n|$)', obs)
         or re.search(r'Instruction: \[SEP\] (.+?) \[SEP\]', obs))
    return m.group(1).strip() if m else obs.split('\n')[0].strip()


class Run:
    """One ReAct-style loop against an env instance; keeps counters across calls (for ADaPT/Reflexion)."""

    def __init__(self, a):
        self.a, self.steps, self.calls, self.ptok, self.ctok, self.actions = a, 0, 0, 0, 0, []
        self.reward, self.done, self.overflow = 0, False, False

    def chat(self, msgs):
        try:
            text, u = E.chat(self.a.llm_url, msgs, self.a.seed)
        except E.ContextFull:
            self.overflow = True
            return None
        self.calls += 1; self.ptok += u['prompt_tokens']; self.ctok += u['completion_tokens']
        return text

    def loop(self, env, msgs, budget, decorate=lambda o: o, stop_word=None):
        """Run until env done, budget spent, context full, or the model says a stop word. Returns (said stop_word[0], transcript)."""
        lines = []
        while budget > 0 and not self.done and self.steps < self.a.max_steps:
            text = self.chat(msgs)
            if text is None:
                break
            msgs.append(dict(role='assistant', content=text))
            act = parse(self.a.env, text)
            if not self.a.env.startswith('textcraft'):
                act = E.normalize_action(act, env)  # R9: no-op unless ACTION_NORM=1
            if stop_word and act is not None and act.strip().lower().strip('.') in stop_word:
                return act.strip().lower().startswith(stop_word[0]), lines
            if act is None:
                obs = "Error: Only one 'Action' is allowed per response. Please adjust your response."
            else:
                self.actions.append(act)
                obs, self.reward, self.done = env.step(act)
                self.steps += 1; budget -= 1
            lines.append(f'> {act}\n{obs}')
            msgs.append(dict(role='user', content=decorate(obs)))
        return False, lines

    def result(self, idx, env, **extra):
        return dict(id=idx, success=bool(env.success(self.reward)), score=self.reward, steps=self.steps,
                    actions=self.actions, llm_calls=self.calls, prompt_tokens=self.ptok, completion_tokens=self.ctok, **extra)


def stateact(idx, a):
    env = E.ENVS[a.env](); obs = env.reset(idx); goal = goal_of(a.env, obs); r = Run(a)
    rule = ('\n\nIn every response, first restate the goal and track the state, in exactly this format:\n'
            'Goal: <the task goal>\nCurrent Location: <where you are>\nCurrent Inventory: <what you are holding>\n'
            'Thought: <your reasoning>\nAction: <your next action>')
    msgs = [dict(role='user', content=E.intro(a.env) + rule), dict(role='assistant', content=OK), dict(role='user', content=obs)]
    r.loop(env, msgs, a.max_steps, decorate=lambda o: f'{o}\nRemember, your goal is: {goal}')
    env.close()
    return r.result(idx, env, overflow=r.overflow)


def reflexion(idx, a, trials=2):
    memory, out = [], None
    tot = dict(steps=0, calls=0, ptok=0, ctok=0)
    for t in range(trials):
        env = E.ENVS[a.env](); obs = env.reset(idx); r = Run(a)
        # reflections go with the task (as in Reflexion), not into the format instructions, which the model tends to echo
        mem = ('\n\nYour reflections on previous failed attempts at this same task:\n' + '\n'.join(f'- {m}' for m in memory)) if memory else ''
        msgs = [dict(role='user', content=E.intro(a.env)), dict(role='assistant', content=OK), dict(role='user', content=obs + mem)]
        _, lines = r.loop(env, msgs, a.max_steps)
        for k in tot:
            tot[k] += getattr(r, k)
        out = r.result(idx, env, trials_used=t + 1)
        env.close()
        if out['success'] or t == trials - 1:
            break
        q = (E.intro(a.env) + '\n\nYou attempted the task below and failed.\n' + obs[:1500] + '\n\nYour trajectory (last steps):\n'
             + '\n'.join(lines[-25:])[-5000:] + '\n\nIn two or three sentences, diagnose why you failed and state a concrete plan '
             'for your next attempt. Do not output an Action.')
        refl = Run(a).chat([dict(role='user', content=q)]) or ''
        tot['calls'] += 1
        memory.append(refl.strip().replace('\n', ' ')[:600])
    out.update(llm_calls=tot['calls'], prompt_tokens=tot['ptok'], completion_tokens=tot['ctok'], steps_all_trials=tot['steps'],
               reflections=memory)
    return out


def adapt(idx, a, max_depth=3):
    env = E.ENVS[a.env](); obs = env.reset(idx); r = Run(a); goal = goal_of(a.env, obs)
    k = max(5, a.max_steps // 3)  # executor budget per (sub)task
    hist = []

    def execute(task, depth):
        rule = (f'\n\nYour current task is: {task}\nWhen you believe this task is complete, respond with "Action: done". '
                'If you are sure you cannot complete it, respond with "Action: fail".')
        msgs = [dict(role='user', content=E.intro(a.env) + rule), dict(role='assistant', content=OK),
                dict(role='user', content=obs + ('\n\nWhat happened so far:\n' + '\n'.join(hist[-12:])[-4000:] if hist else ''))]
        said_done, lines = r.loop(env, msgs, k, stop_word=('done', 'fail'))
        hist.extend(lines)
        if r.done or depth == 1:  # the top-level task counts as done only on real env success (never cut the episode short)
            return bool(r.done and env.success(r.reward))
        return said_done

    def plan(task):
        q = (E.intro(a.env) + f'\n\nThe task "{task}" was too hard to do in one go.\n' + obs[:1500]
             + '\n\nWhat happened so far:\n' + '\n'.join(hist[-12:])[-4000:]
             + '\n\nDecompose the task into at most 4 simpler sub-tasks that must all be done in order. '
             'Answer as a numbered list, one sub-task per line.')
        text = r.chat([dict(role='user', content=q)]) or ''
        subs = [re.sub(r'^\s*\d+[.)]\s*', '', l).strip() for l in text.splitlines() if re.match(r'^\s*\d+[.)]', l)]
        return [s for s in subs if s][:4]

    def solve(task, depth):
        if r.done or r.steps >= a.max_steps or r.overflow:
            return env.success(r.reward)
        if execute(task, depth):
            return True
        if depth >= max_depth:
            return False
        subs = plan(task)
        if not subs:
            return False
        return all(solve(s, depth + 1) for s in subs)

    solve(goal, 1)
    env.close()
    return r.result(idx, env, overflow=r.overflow)


_EXPEL = {}


def expel_memory(a, n_train=40):
    """Gather: ReAct on n_train TRAIN tasks (fixed seed); Distill: insights from success/failure; cached on disk."""
    path = f'{E.R}/runs/baselines/expel_memory_{a.env}' + os.environ.get('EXPEL_TAG', '') + '.json'  # EXPEL_TAG: per-backbone memory (A1)
    if a.env in _EXPEL:
        return _EXPEL[a.env]
    if os.path.exists(path):
        _EXPEL[a.env] = json.load(open(path)); return _EXPEL[a.env]
    train = E.ids(a.env, 'train')
    pick = sorted(random.Random(20261104).sample(train, min(n_train, len(train))))
    trajs = []
    for i in pick:
        env = E.ENVS[a.env](); obs = env.reset(i); r = Run(a)
        msgs = [dict(role='user', content=E.intro(a.env)), dict(role='assistant', content=OK), dict(role='user', content=obs)]
        _, lines = r.loop(env, msgs, a.max_steps)
        trajs.append(dict(id=i, task=obs[:1500], success=bool(env.success(r.reward)), lines=lines[-20:]))
        env.close()
    succ = [t for t in trajs if t['success']]; fail = [t for t in trajs if not t['success']]
    show = lambda t: ('SUCCESS' if t['success'] else 'FAILURE') + '\nTask: ' + t['task'][:600] + '\n' + '\n'.join(t['lines'])[-1500:]
    sample = (succ[:3] + fail[:3]) or trajs[:4]
    q = (E.intro(a.env) + '\n\nHere are past attempts at tasks in this environment:\n\n' + '\n\n'.join(show(t) for t in sample)
         + '\n\nBy comparing successes and failures, write at most 8 short, general insights (rules of thumb) that would '
         'help solve new tasks in this environment. One per line, starting with "- ".')
    text = Run(a).chat([dict(role='user', content=q)]) or ''
    insights = [l[2:].strip() for l in text.splitlines() if l.strip().startswith('- ')][:8]
    mem = dict(insights=insights, successes=succ, n_train=len(pick), n_success=len(succ))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(mem, open(path, 'w'), indent=1)
    _EXPEL[a.env] = mem
    return mem


def expel(idx, a):
    mem = expel_memory(a)
    env = E.ENVS[a.env](); obs = env.reset(idx); r = Run(a)
    w = E.words(obs[:1500])
    near = sorted(mem['successes'], key=lambda t: (-len(w & E.words(t['task'])) / max(1, len(w | E.words(t['task']))), t['id']))[:2]
    extra = ''
    if mem['insights']:
        extra += '\n\nInsights from past experience:\n' + '\n'.join(f'- {x}' for x in mem['insights'])
    for t in near:
        extra += '\n\nA successful past episode:\nTask: ' + t['task'][:800] + '\n' + '\n'.join(t['lines'])[-1500:]
    # R3 fix (2026-11-15): insights/exemplars go with the task, not after the format instructions, which the model echoed
    # into its Action (81% multi-line actions on ScienceWorld test); same placement fix as Reflexion in f537359
    msgs = [dict(role='user', content=E.intro(a.env)), dict(role='assistant', content=OK), dict(role='user', content=obs + extra)]
    r.loop(env, msgs, a.max_steps)
    env.close()
    return r.result(idx, env, overflow=r.overflow)


E.METHODS.update(stateact=stateact, reflexion=reflexion, adapt=adapt, expel=expel)

if __name__ == '__main__':
    E.main()
