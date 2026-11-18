"""PinPatch training (TRAIN split only) and frozen evaluation.

usage: python -I run.py train            -> learns model on TRAIN, logs learning curve, saves model.pkl
       python -I run.py eval SPLIT [ARM]  -> frozen evaluation, model never updated across episodes
"""
import sys, json, pickle, random, time, math
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinpatch_tc.paths import WORK, DATA
from collections import Counter
from pinpatch_tc.envwrap import Episode, parse_inventory
from pinpatch_tc.agent import LiftedModel, Planner, parse_commands, parse_action, substitution_of, analogize

D = WORK
H = 50


def load(split):
    return json.load(open(DATA + f'{split}.json'))


def menu(cmds, inv, rng):
    """Exploration action space: visible commands, gets of mentioned entities, and
    visible commands with one ingredient replaced by a held item."""
    acts = []
    ents = set()
    for line, (verb, slots) in cmds:
        acts.append(line)
        for role, n, e in slots:
            ents.add((n, e))
    for n, e in ents:
        acts.append(f'get {n} {e}')
    if inv:
        for line, (verb, slots) in cmds:
            for i, (role, n, e) in enumerate(slots):
                if role == 1:
                    s = rng.choice(sorted(inv))
                    new = list(slots); new[i] = (role, n, s)
                    from pinpatch_tc.agent import render
                    acts.append(render(verb, new))
    return acts


def run_episode(inst, model, rng, learn=False, explore=0.0, random_policy=False,
                use_substitution=True, online=True, literal_menu=False, use_memory=False, memory_gate=False,
                llm_prior=False, analogy=False):
    ep = Episode(inst, horizon=H)
    cmds, goal = parse_commands(ep.observation)
    stats = Counter()
    planner = None
    while not ep.done:
        inv = parse_inventory(ep.inventory_text())
        if planner is None or learn:
            # rebuild so freshly learned coefficients are reflected (learning mode)
            old = planner
            planner = Planner(model, cmds, goal, rng, explore=explore, use_substitution=use_substitution,
                              use_memory=use_memory, memory_gate=memory_gate)
            if old is not None:
                planner.get_failed, planner.bad_pairs, planner.bad_actions = old.get_failed, old.bad_pairs, old.bad_actions
        plan = None if random_policy else planner.plan(inv, H - ep.steps)
        if plan:
            action = plan[0]; stats['planned'] += 1
        else:
            action = rng.choice(menu(cmds, inv, rng)); stats['fallback'] += 1
        if literal_menu:
            legal = {l for l, _ in cmds} | {a for a in menu(cmds, {}, rng) if a.startswith('get')}
            if action not in legal:
                action = rng.choice(sorted(legal)); stats['menu_blocked'] += 1
        obs, r, done, info = ep.step(action)
        inv2 = parse_inventory(ep.inventory_text()) if not done else inv
        sub = substitution_of(action, cmds)
        if learn:
            model.observe(action, inv, inv2, not info['invalid'], planner.ctx, sub)
        if online and info['invalid']:
            p = parse_action(action)
            planner.bad_actions[action] += 1
            if p and p[0] == 'get':
                planner.get_failed.add(p[1][0][2])
                e = p[1][0][2]
                if analogy and not planner.producers.get(e) and not planner.memory_producers.get(e):
                    planner.memory_producers[e] = analogize(e, model)
                if (llm_prior and PRIOR and len(e.split()) > 1 and not planner.producers.get(e)
                        and not planner.memory_producers.get(e)):
                    stats['llm_calls'] += 1
                    planner.memory_producers[e] = PRIOR(e, model)
            elif p and any(action == c[0] for v in planner.memory_producers.values() for c in v):
                planner.bad_actions[action] += 1  # an external candidate gets one try only
            if sub:
                consumed_ok = all(inv.get(e, 0) >= n for role, n, e in p[1] if role == 1)
                if consumed_ok:
                    for g, s in sub.items():
                        planner.bad_pairs.add((g, s))
    stats['invalid'] = ep.invalid
    return {'id': inst['id'], 'goal': goal, 'won': ep.won, 'steps': ep.steps, **stats}


def train(seed=0, rounds=4, n_random=60, tag=''):
    rng = random.Random(seed)
    tr = load('train')
    model = LiftedModel(); curve = []; t0 = time.time()
    # phase 0: uninformed random exploration (bootstrap the effect model)
    res = [run_episode(x, model, rng, learn=True, random_policy=True) for x in tr[:n_random]]
    curve.append({'round': 0, 'policy': 'random', 'success': sum(r['won'] for r in res) / len(res), 'n': len(res)})
    model.fit_grounding()
    # rounds of model-based RL: plan with epsilon-exploration on uncertain choices, refit
    for k in range(1, rounds + 1):
        eps = max(0.0, 0.3 - 0.1 * (k - 1))
        order = list(tr); rng.shuffle(order)
        res = []
        for x in order:
            res.append(run_episode(x, model, rng, learn=True, explore=eps, use_memory=MEM))
        model.fit_grounding()
        curve.append({'round': k, 'policy': f'planner eps={eps:.1f}', 'success': sum(r['won'] for r in res) / len(res),
                      'n': len(res), 'fallback_actions': sum(r.get('fallback', 0) for r in res)})
        print(curve[-1], flush=True)
    pickle.dump(model, open(D + f'model{tag}.pkl', 'wb'))
    json.dump({'curve': curve, 'model': model.to_json(), 'seconds': time.time() - t0}, open(D + f'train_log{tag}.json', 'w'), indent=1)
    return model


MEM = False
PRIOR = None  # callable(entity, model) -> candidate producers; set by the caller
ARMS = {
    'pinpatch_tc': dict(),
    'mem_gated': dict(use_memory=True, memory_gate=True),
    'llm_gated': dict(memory_gate=True, llm_prior=True),
    'mem_llm_gated': dict(use_memory=True, memory_gate=True, llm_prior=True),
    'mem_analog_gated': dict(use_memory=True, memory_gate=True, analogy=True),
    'mem_analog_llm_gated': dict(use_memory=True, memory_gate=True, analogy=True, llm_prior=True),
    'pinpatch_tc_mem_v2_posthoc': dict(use_memory=True),
    'no_substitution': dict(use_substitution=False),
    'no_online_correction': dict(online=False),
    'literal_menu_031': dict(literal_menu=True, use_substitution=False),
}


def evaluate(split, arm='pinpatch_tc', seed=0, tag=''):
    model = pickle.load(open(D + f'model{tag}.pkl', 'rb'))
    frozen = json.dumps(model.to_json(), sort_keys=True)
    rng = random.Random(seed)
    res = [run_episode(x, model, rng, learn=False, **ARMS[arm]) for x in load(split)]
    assert json.dumps(model.to_json(), sort_keys=True) == frozen  # frozen across episodes
    s = sum(r['won'] for r in res)
    out = {'split': split, 'arm': arm, 'success': s, 'n': len(res), 'rate': s / len(res),
           'mean_steps_success': sum(r['steps'] for r in res if r['won']) / max(1, s),
           'invalid_actions': sum(r['invalid'] for r in res), 'fallback_actions': sum(r.get('fallback', 0) for r in res),
           'planned_actions': sum(r.get('planned', 0) for r in res), 'episodes': res}
    json.dump(out, open(D + f'results/{split}_{arm}{tag}.json', 'w'), indent=1)
    print({k: v for k, v in out.items() if k != 'episodes'}, flush=True)
    return out


if __name__ == '__main__':
    # determinism: set iteration order depends on the hash seed, so pin it (re-exec once)
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable] + sys.argv)
    import os; os.makedirs(D + 'results', exist_ok=True)
    if sys.argv[1] == 'train':
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        v2 = len(sys.argv) > 3 and sys.argv[3] == 'v2'
        MEM = v2
        train(seed=sd, tag=('_v2' if v2 else '') + ('' if sd == 0 else f'_s{sd}'))
    else:
        sd = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        arm = sys.argv[3] if len(sys.argv) > 3 else 'pinpatch_tc'
        evaluate(sys.argv[2], arm, tag=('_v2' if 'v2' in arm else '') + ('' if sd == 0 else f'_s{sd}'))
