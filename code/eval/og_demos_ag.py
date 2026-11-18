"""OntoGuard demonstrations on our AgentGym environments (TRAIN only), in the format of the MetaAct repo's
scripts/ontoguard_build.py --stage demos: {id, task_type, goal, obs0, steps: [(action, obs)], won}.
The repo reads gold paths from env internals AgentGym does not expose; equivalents here:
  sciworld  - replay the AgentGym train expert demos (data/splits/sciworld_demos.json), round-robin over task types
  textcraft - the repo's own crafting-tree solver (demos_tc), run on our env
  webshop   - the repo's demos_ws on the WebShop human IL trajectories, excluding our dev/test instructions
  alfworld  - built separately in the ALFWorld env (og_demos_alf.py), expert plan from textworld
usage: og_demos_ag.py ENV N OUT"""
import json, os, random, re, sys

import metaact_ag as M
import run_eval as E


def sw(n):
    demos = json.load(open(f'{E.R}/data/splits/sciworld_demos.json'))
    games = json.load(open(f'{E.R}/data/splits/sciworld_games.json'))
    train = set(json.load(open(f'{E.R}/data/splits/sciworld.json'))['train'])
    by = {}
    for d in sorted(demos, key=lambda d: int(d['id'])):
        if int(d['id']) in train:
            by.setdefault(games[str(d['id'])], []).append(d)
    pick, k = [], 0
    while len(pick) < n and any(len(v) > k for v in by.values()):
        pick += [v[k] for t, v in sorted(by.items()) if len(v) > k][: n - len(pick)]; k += 1
    out = []
    for d in pick:
        env = M.AGEnv('sciworld'); env.max_steps = 1000
        acts = d['actions'] if isinstance(d['actions'], list) else eval(d['actions'])
        obs0, info = env.reset(dict(ag_idx=int(d['id'])))
        steps, won = [], False
        for a in acts:
            o, _, done, si = env.step(a); steps.append((a, o)); won = won or si['won']
            if done:
                break
        env.close()
        out.append(dict(id=f"sciworld_{d['id']}", task_type=games[str(d['id'])], goal=info['goal'], obs0=obs0,
                        steps=steps, won=bool(won), score=env.score))
        print('  sw demo', d['id'], won, len(steps), flush=True)
    return out


def tc(n):
    import ontoguard_build as OB
    sys.path.insert(0, M.MA + '/third_party')
    tasks = [json.loads(l) for l in open(M.tasks_file('textcraft', f'train{n}'))]

    class TCEnv(M.AGEnv):  # the repo solver reads env._env.{goal, inventory, crafting_tree} and env._won
        def __init__(self, cfg=None):
            super().__init__('textcraft'); self.max_steps = 1000
            self._env = self.env.env  # AgentGym TextCraftEnv: goal, inventory, crafting_tree
    import metaact.envs.textcraft_env as te
    te.TextCraftEnv = TCEnv
    return OB.demos_tc(tasks)


def ws(n):
    import ontoguard_build as OB
    excl = set()
    sp = json.load(open(f'{E.R}/data/splits/webshop.json'))
    env = E.WebShop()
    for i in sp['test'] + sp['dev']:
        excl.add(OB.norm(E.ws_instruction(env.reset(i))))
    env.close()
    os.environ['WEBSHOP_ROOT'] = f'{E.R}/data/webshop/il_root'
    os.makedirs(f'{E.R}/data/webshop/il_root/baseline_models/data', exist_ok=True)
    link = f'{E.R}/data/webshop/il_root/baseline_models/data/il_trajs_finalized_images.zip'
    if not os.path.exists(link):
        os.symlink(f'{E.R}/data/webshop/il/il_trajs_finalized_images.zip', link)
    demos, n_ex = OB.demos_ws(n=n + len(excl))  # same shuffled order as the repo; our dev/test removed, then first n
    demos = [d for d in demos if OB.norm(d.get('goal', '')) not in excl][:n]
    print('ws demos', len(demos), 'repo exclusions', n_ex, 'our dev/test instructions', len(excl))
    return demos


if __name__ == '__main__':
    envname, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    M.patch(); sys.path.insert(0, M.MA + '/scripts'); os.chdir(M.MA)
    demos = dict(sciworld=sw, textcraft=tc, webshop=ws)[envname](n)
    ok = [d for d in demos if d.get('won')]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        for d in ok:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'[{envname}] demos={len(demos)} won={len(ok)} -> {out}')
